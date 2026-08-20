from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from rankflow import mm_tensor, naive_factors


torch.set_default_dtype(torch.float64)


@dataclass
class SpectrumInfo:
    rows: int
    cols: int
    rank: int
    nullity: int
    sigma_max: float
    sigma_min_positive: float
    condition_positive: float
    cutoff: float


@dataclass
class CorrectorInfo:
    converged: bool
    iterations: int
    residual_norm: float
    step_norm: float


def pack(U: torch.Tensor, V: torch.Tensor, W: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.cat((U.reshape(-1), V.reshape(-1), W.reshape(-1), a.reshape(-1)))


def unpack(theta: torch.Tensor, n: int, rank: int):
    d = n * n
    block = d * rank
    U = theta[:block].reshape(d, rank)
    V = theta[block : 2 * block].reshape(d, rank)
    W = theta[2 * block : 3 * block].reshape(d, rank)
    a = theta[3 * block : 3 * block + rank]
    return U, V, W, a


def amp_index(n: int, rank: int, channel: int) -> int:
    return 3 * n * n * rank + channel


def unit_columns(X: torch.Tensor) -> torch.Tensor:
    # These columns never intentionally approach zero in this toy.  clamp_min
    # keeps the Jacobian finite if an exploratory step gets silly.
    return X / X.norm(dim=0, keepdim=True).clamp_min(1.0e-14)


def reconstruct(theta: torch.Tensor, n: int, rank: int) -> torch.Tensor:
    U, V, W, a = unpack(theta, n, rank)
    U = unit_columns(U)
    V = unit_columns(V)
    W = unit_columns(W)
    return torch.einsum("ir,jr,kr,r->ijk", U, V, W, a)


def residual_vector(theta: torch.Tensor, target: torch.Tensor, n: int, rank: int) -> torch.Tensor:
    return (reconstruct(theta, n, rank) - target).reshape(-1)


def naive_theta(n: int, device: str) -> torch.Tensor:
    U, V, W = naive_factors(n, device)
    a = torch.ones(n**3, device=device)
    return pack(U, V, W, a)


def jacobian(theta: torch.Tensor, target: torch.Tensor, n: int, rank: int) -> torch.Tensor:
    x = theta.detach().clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(
        lambda q: residual_vector(q, target, n, rank),
        x,
        vectorize=True,
    )
    return J.detach()


def robust_svd(A: torch.Tensor, *, full_matrices: bool = True):
    """Numerically defensive SVD for the ill-conditioned Jacobians in this toy.

    PyTorch's CPU SVD normally uses a divide-and-conquer LAPACK driver.  Very
    repeated/ill-conditioned spectra can occasionally make that driver fail to
    converge even when the matrix is finite.  We first scale the matrix to O(1)
    and retry PyTorch; if that still fails, fall back to SciPy's conservative
    ``gesvd`` driver.  Singular values are returned in the original scale.

    The fallback deliberately happens on CPU/float64.  These Jacobians are small
    enough that robustness matters much more than device locality.
    """
    if A.numel() == 0:
        k = min(A.shape)
        U = torch.empty((A.shape[0], A.shape[0] if full_matrices else k), dtype=A.dtype, device=A.device)
        S = torch.empty((k,), dtype=A.dtype, device=A.device)
        Vh = torch.empty((A.shape[1] if full_matrices else k, A.shape[1]), dtype=A.dtype, device=A.device)
        return U, S, Vh
    if not bool(torch.isfinite(A).all()):
        raise torch._C._LinAlgError("robust_svd received non-finite matrix")

    scale = float(A.detach().abs().max())
    if not math.isfinite(scale) or scale == 0.0:
        scale = 1.0
    B = (A / scale).contiguous()

    try:
        U, S, Vh = torch.linalg.svd(B, full_matrices=full_matrices)
        return U, S * scale, Vh
    except (torch._C._LinAlgError, RuntimeError):
        pass

    # ``gesvd`` is slower than divide-and-conquer ``gesdd`` but substantially
    # more forgiving of clustered singular values.  Keep this optional so the
    # original Torch-only dependency still works on ordinary states.
    try:
        import numpy as np
        import scipy.linalg

        Bcpu = B.detach().to(device="cpu", dtype=torch.float64).numpy()
        Un, Sn, Vhn = scipy.linalg.svd(
            Bcpu,
            full_matrices=full_matrices,
            lapack_driver="gesvd",
            check_finite=False,
        )
        U = torch.from_numpy(np.asarray(Un)).to(dtype=A.dtype, device=A.device)
        S = torch.from_numpy(np.asarray(Sn)).to(dtype=A.dtype, device=A.device) * scale
        Vh = torch.from_numpy(np.asarray(Vhn)).to(dtype=A.dtype, device=A.device)
        return U, S, Vh
    except Exception as exc:
        raise torch._C._LinAlgError(
            "SVD failed with both scaled torch.linalg.svd and scipy gesvd"
        ) from exc


def svd_info(J: torch.Tensor, rcond: float = 1.0e-10, full: bool = True):
    U, S, Vh = robust_svd(J, full_matrices=full)
    sigma_max = float(S[0]) if len(S) else 0.0
    cutoff = rcond * sigma_max if sigma_max else rcond
    numerical_rank = int((S > cutoff).sum())
    nullity = J.shape[1] - numerical_rank
    if numerical_rank:
        sigma_min = float(S[numerical_rank - 1])
        cond = sigma_max / sigma_min
    else:
        sigma_min = 0.0
        cond = math.inf
    info = SpectrumInfo(
        rows=J.shape[0],
        cols=J.shape[1],
        rank=numerical_rank,
        nullity=nullity,
        sigma_max=sigma_max,
        sigma_min_positive=sigma_min,
        condition_positive=cond,
        cutoff=cutoff,
    )
    return U, S, Vh, info


def tangent_projection(v: torch.Tensor, J: torch.Tensor, rcond: float = 1.0e-10):
    """Project parameter-space vector v onto ker(J)."""
    _, S, Vh, info = svd_info(J, rcond=rcond, full=True)
    # Vh is square when full_matrices=True. Rows [rank:] span ker(J).
    N = Vh[info.rank :, :].T
    if N.shape[1] == 0:
        return torch.zeros_like(v), info
    return N @ (N.T @ v), info


def killability(theta: torch.Tensor, J: torch.Tensor, n: int, rank: int, rcond: float):
    """
    For channel r, p = P_ker(J) e_r.

    ||p|| lies in [0,1] and measures how much the channel amplitude can move
    while preserving the tensor to first order.  The minimum-norm tangent
    direction that changes that amplitude at unit rate has norm 1/||p||.
    """
    _, _, Vh, info = svd_info(J, rcond=rcond, full=True)
    N = Vh[info.rank :, :].T
    _, _, _, a = unpack(theta, n, rank)
    rows = []

    for channel in range(rank):
        e = torch.zeros(J.shape[1], dtype=J.dtype, device=J.device)
        e[amp_index(n, rank, channel)] = 1.0
        p = N @ (N.T @ e) if N.shape[1] else torch.zeros_like(e)
        score = float(p.norm())
        rows.append(
            {
                "channel": channel,
                "amplitude": float(a[channel]),
                "killability": score,
                "unit_change_cost": (1.0 / score) if score > 1.0e-14 else math.inf,
            }
        )
    return rows, info


def pinv_solve(A: torch.Tensor, b: torch.Tensor, rcond: float = 1.0e-10) -> torch.Tensor:
    """Minimum-norm least-squares solution A x = b using a truncated SVD."""
    if A.numel() == 0:
        return torch.zeros(A.shape[1], dtype=A.dtype, device=A.device)
    U, S, Vh = robust_svd(A, full_matrices=False)
    if len(S) == 0:
        return torch.zeros(A.shape[1], dtype=A.dtype, device=A.device)
    cutoff = rcond * float(S[0])
    keep = S > cutoff
    if not bool(keep.any()):
        return torch.zeros(A.shape[1], dtype=A.dtype, device=A.device)
    coeff = (U[:, keep].T @ b) / S[keep]
    return Vh[keep, :].T @ coeff


def gauss_newton_correct(
    theta: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    *,
    frozen: set[int] | None = None,
    tol: float = 1.0e-11,
    max_iters: int = 12,
    rcond: float = 1.0e-10,
) -> tuple[torch.Tensor, CorrectorInfo]:
    """Project a nearby point back toward F(theta)=0 with damped GN steps."""
    x = theta.detach().clone()
    frozen = frozen or set()
    free_mask = torch.ones(x.numel(), dtype=torch.bool, device=x.device)
    for idx in frozen:
        free_mask[idx] = False

    total_step = 0.0
    previous = math.inf

    for iteration in range(max_iters + 1):
        r = residual_vector(x, target, n, rank).detach()
        rn = float(r.norm())
        if rn <= tol:
            return x, CorrectorInfo(True, iteration, rn, total_step)
        if iteration == max_iters:
            break

        J = jacobian(x, target, n, rank)
        Jf = J[:, free_mask]
        delta_free = pinv_solve(Jf, -r, rcond=rcond)
        if float(delta_free.norm()) == 0.0:
            break

        delta = torch.zeros_like(x)
        delta[free_mask] = delta_free

        # Small line search. Near the manifold alpha=1 is normally accepted;
        # farther away this prevents a pseudoinverse step from exploding.
        accepted = False
        alpha = 1.0
        for _ in range(10):
            candidate = x + alpha * delta
            cr = float(residual_vector(candidate, target, n, rank).norm())
            if cr < rn:
                x = candidate
                total_step += alpha * float(delta.norm())
                previous = cr
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    rn = float(residual_vector(x, target, n, rank).norm())
    return x, CorrectorInfo(rn <= tol, max_iters, rn, total_step)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def geometry_report(*, n: int, out: Path, device: str, rcond: float):
    rank = n**3
    target = mm_tensor(n, device)
    theta = naive_theta(n, device)
    r0 = float(residual_vector(theta, target, n, rank).norm())
    J = jacobian(theta, target, n, rank)
    _, S, _, info = svd_info(J, rcond=rcond, full=True)
    kills, _ = killability(theta, J, n, rank, rcond)

    spectrum_rows = [
        {
            "index": i,
            "singular_value": float(s),
            "above_cutoff": bool(float(s) > info.cutoff),
        }
        for i, s in enumerate(S)
    ]
    write_csv(out / "geometry_spectrum.csv", spectrum_rows)
    write_csv(out / "geometry_killability.csv", kills)

    payload = {
        "n": n,
        "rank": rank,
        "initial_residual_norm": r0,
        "spectrum": asdict(info),
        "killability": kills,
    }
    with (out / "geometry.json").open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"initial residual norm: {r0:.3e}")
    print(
        f"J shape={info.rows}x{info.cols} rank={info.rank} nullity={info.nullity} "
        f"sigma+=[{info.sigma_min_positive:.3e}, {info.sigma_max:.3e}] "
        f"cond+={info.condition_positive:.3e}"
    )
    for row in kills:
        cost = row["unit_change_cost"]
        cost_text = f"{cost:.3e}" if math.isfinite(cost) else "inf"
        print(
            f"channel {row['channel']}: a={row['amplitude']:+.6f} "
            f"killability={row['killability']:.3e} cost={cost_text}"
        )



def remove_normalization_gauge(theta: torch.Tensor, q: torch.Tensor, n: int, rank: int) -> torch.Tensor:
    """Remove useless radial motions of normalized CP factor columns."""
    U, V, W, _ = unpack(theta, n, rank)
    qU, qV, qW, qa = unpack(q.clone(), n, rank)
    for X, Q in ((U, qU), (V, qV), (W, qW)):
        Xhat = unit_columns(X)
        Q -= Xhat * (Xhat * Q).sum(dim=0, keepdim=True)
    return pack(qU, qV, qW, qa)


def escape_search(
    theta: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    *,
    size: float,
    trials: int,
    seed: int,
    rcond: float,
    tol: float,
    out: Path,
) -> torch.Tensor:
    """
    Move sideways from a symmetric/singular exact algorithm before trying to
    kill a channel.  Random directions are sampled in ker(J), radial CP gauge
    motion is removed, each candidate is corrected back to F=0, and we keep
    the exact point with the largest post-move channel killability.

    This is intentionally a tiny stochastic probe of second-order curvature,
    not a claim that random search is the final continuation strategy.
    """
    if size <= 0 or trials <= 0:
        return theta

    torch.manual_seed(seed)
    J = jacobian(theta, target, n, rank)
    _, _, Vh, info = svd_info(J, rcond=rcond, full=True)
    N = Vh[info.rank :, :].T
    if N.shape[1] == 0:
        return theta

    rows = []
    best_theta = theta
    best_score = -1.0

    for trial_idx in range(trials):
        q = N @ torch.randn(N.shape[1], dtype=theta.dtype, device=theta.device)
        q = remove_normalization_gauge(theta, q, n, rank)
        q, _ = tangent_projection(q, J, rcond=rcond)
        q = remove_normalization_gauge(theta, q, n, rank)
        qn = float(q.norm())
        if qn <= 1.0e-12:
            continue
        q = q / qn

        predictor = theta + size * q
        corrected, cinfo = gauss_newton_correct(
            predictor, target, n, rank, tol=tol, max_iters=25, rcond=rcond
        )
        if not cinfo.converged:
            rows.append({
                "trial": trial_idx, "converged": False, "max_killability": 0.0,
                "best_channel": -1, "distance": float((predictor-theta).norm()),
                "sigma_min_positive": 0.0, "jacobian_rank": -1,
                "min_amplitude": math.nan, "max_amplitude": math.nan,
            })
            continue

        Jc = jacobian(corrected, target, n, rank)
        kills, infoc = killability(corrected, Jc, n, rank, rcond)
        best_row = max(kills, key=lambda row: row["killability"])
        _, _, _, ac = unpack(corrected, n, rank)
        score = float(best_row["killability"])
        rows.append({
            "trial": trial_idx, "converged": True, "max_killability": score,
            "best_channel": int(best_row["channel"]),
            "distance": float((corrected-theta).norm()),
            "sigma_min_positive": infoc.sigma_min_positive,
            "jacobian_rank": infoc.rank,
            "min_amplitude": float(ac.min()), "max_amplitude": float(ac.max()),
        })
        if score > best_score:
            best_score = score
            best_theta = corrected

    write_csv(out / "escape_trials.csv", rows)
    if best_score >= 0:
        print(f"escape search: best killability={best_score:.3e} from {trials} trials at size={size:g}")
    return best_theta

def tangent_kick(
    theta: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    *,
    size: float,
    seed: int,
    rcond: float,
    tol: float,
) -> torch.Tensor:
    if size <= 0:
        return theta
    torch.manual_seed(seed)
    J = jacobian(theta, target, n, rank)
    q = torch.randn_like(theta)
    tangent, _ = tangent_projection(q, J, rcond=rcond)
    tn = float(tangent.norm())
    if tn == 0.0:
        return theta
    predictor = theta + size * tangent / tn
    corrected, info = gauss_newton_correct(
        predictor,
        target,
        n,
        rank,
        tol=tol,
        max_iters=20,
        rcond=rcond,
    )
    if not info.converged:
        raise RuntimeError(
            f"tangent kick could not be corrected: residual={info.residual_norm:.3e}"
        )
    return corrected


def manifold_flow(
    *,
    n: int,
    out: Path,
    device: str,
    seed: int,
    target_channel: int,
    amp_step: float,
    max_steps: int,
    min_amp_step: float,
    rcond: float,
    tol: float,
    kick: float,
    escape_size: float,
    escape_trials: int,
):
    """
    Follow F(theta)=0 while decreasing one channel amplitude.

    Predictor:
      choose the minimum-norm tangent direction with da_target/ds = -sign(a).

    Corrector:
      Gauss-Newton back to F=0 while freezing the target amplitude at its
      predicted value.

    If the projected amplitude coordinate becomes tiny, the required tangent
    speed blows up. That is exactly the local signature of an amplitude fold /
    loss of first-order killability.
    """
    rank = n**3
    target = mm_tensor(n, device)
    theta = naive_theta(n, device)
    theta = tangent_kick(
        theta,
        target,
        n,
        rank,
        size=kick,
        seed=seed,
        rcond=rcond,
        tol=tol,
    )
    theta = escape_search(
        theta,
        target,
        n,
        rank,
        size=escape_size,
        trials=escape_trials,
        seed=seed,
        rcond=rcond,
        tol=tol,
        out=out,
    )

    J0 = jacobian(theta, target, n, rank)
    kills0, _ = killability(theta, J0, n, rank, rcond)
    if target_channel < 0:
        target_channel = max(kills0, key=lambda row: row["killability"])["channel"]
    if not (0 <= target_channel < rank):
        raise ValueError(f"target channel must be in [0,{rank})")

    print(f"following channel {target_channel}; initial amp step={amp_step:g}")
    rows: list[dict] = []
    current_step = amp_step
    status = "max_steps"

    for step in range(max_steps):
        r = residual_vector(theta, target, n, rank).detach()
        J = jacobian(theta, target, n, rank)
        _, S, Vh, info = svd_info(J, rcond=rcond, full=True)
        N = Vh[info.rank :, :].T
        _, _, _, a = unpack(theta, n, rank)
        ak = float(a[target_channel])

        e = torch.zeros_like(theta)
        idx = amp_index(n, rank, target_channel)
        e[idx] = 1.0
        p = N @ (N.T @ e) if N.shape[1] else torch.zeros_like(e)
        kill = float(p.norm())
        cost = (1.0 / kill) if kill > 1.0e-14 else math.inf

        rows.append(
            {
                "step": step,
                "amplitude": ak,
                "residual_norm": float(r.norm()),
                "amp_step": current_step,
                "killability": kill,
                "unit_change_cost": cost,
                "jacobian_rank": info.rank,
                "jacobian_nullity": info.nullity,
                "sigma_min_positive": info.sigma_min_positive,
                "sigma_max": info.sigma_max,
                "condition_positive": info.condition_positive,
                "predictor_norm": 0.0,
                "corrector_norm": 0.0,
                "corrector_iters": 0,
                "accepted": True,
            }
        )

        if abs(ak) <= 1.0e-6:
            status = "rank_drop_reached"
            print(f"rank-drop threshold reached at step {step}: a={ak:.3e}")
            break
        if kill <= 1.0e-10:
            status = "amplitude_fold"
            print(f"killability collapsed at step {step}: {kill:.3e}")
            break

        # p = P e and e^T p = ||p||^2. Therefore p/||p||^2 changes the
        # selected amplitude at unit rate while remaining tangent to F=0.
        sign = 1.0 if ak >= 0 else -1.0
        tangent = -sign * p / (kill * kill)

        accepted = False
        trial_step = min(current_step, abs(ak))
        base = theta.clone()
        while trial_step >= min_amp_step:
            predictor_delta = trial_step * tangent
            trial = base + predictor_delta
            desired_a = float(unpack(trial, n, rank)[3][target_channel])

            corrected, cinfo = gauss_newton_correct(
                trial,
                target,
                n,
                rank,
                frozen={idx},
                tol=tol,
                max_iters=16,
                rcond=rcond,
            )
            if cinfo.converged:
                theta = corrected
                accepted = True
                current_step = min(amp_step, trial_step * 1.25)
                rows[-1]["predictor_norm"] = float(predictor_delta.norm())
                rows[-1]["corrector_norm"] = cinfo.step_norm
                rows[-1]["corrector_iters"] = cinfo.iterations
                rows[-1]["accepted"] = True
                print(
                    f"{step:03d} a={ak:+.6f} -> {desired_a:+.6f} "
                    f"kill={kill:.3e} sigmin={info.sigma_min_positive:.3e} "
                    f"|pred|={float(predictor_delta.norm()):.2e} "
                    f"|corr|={cinfo.step_norm:.2e}"
                )
                break
            trial_step *= 0.5

        if not accepted:
            rows[-1]["accepted"] = False
            status = "corrector_failed"
            print(
                f"corrector failed at step {step}; minimum trial amp step "
                f"{trial_step:.3e} < {min_amp_step:.3e}"
            )
            break

    write_csv(out / "manifold_flow.csv", rows)
    torch.save(theta.cpu(), out / "manifold_theta.pt")

    final_r = float(residual_vector(theta, target, n, rank).norm())
    _, _, _, final_a = unpack(theta, n, rank)
    payload = {
        "status": status,
        "target_channel": target_channel,
        "final_residual_norm": final_r,
        "final_amplitudes": [float(x) for x in final_a.cpu()],
        "steps_recorded": len(rows),
        "kick": kick,
        "escape_size": escape_size,
        "escape_trials": escape_trials,
        "seed": seed,
    }
    with (out / "manifold_summary.json").open("w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"status={status}; final residual={final_r:.3e}; "
        f"target amplitude={float(final_a[target_channel]):+.6e}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Jacobian geometry and exact-manifold continuation for matrix multiplication tensors"
    )
    p.add_argument("--mode", choices=["geometry", "manifold", "all"], default="all")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("runs/geometry_flow"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--target-channel",
        type=int,
        default=-1,
        help="channel to kill; -1 selects the largest local killability",
    )
    p.add_argument("--amp-step", type=float, default=0.025)
    p.add_argument("--min-amp-step", type=float, default=1.0e-6)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--rcond", type=float, default=1.0e-10)
    p.add_argument("--tol", type=float, default=1.0e-11)
    p.add_argument(
        "--kick",
        type=float,
        default=0.0,
        help="optional single random tangent displacement, corrected back to the exact manifold",
    )
    p.add_argument(
        "--escape-size",
        type=float,
        default=0.0,
        help="size of sideways tangent probes used to escape the symmetric schoolbook point",
    )
    p.add_argument(
        "--escape-trials",
        type=int,
        default=0,
        help="number of exact sideways probes; keeps the candidate with largest channel killability",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode in ("geometry", "all"):
        geometry_report(n=args.n, out=args.out, device=args.device, rcond=args.rcond)

    if args.mode in ("manifold", "all"):
        manifold_flow(
            n=args.n,
            out=args.out,
            device=args.device,
            seed=args.seed,
            target_channel=args.target_channel,
            amp_step=args.amp_step,
            max_steps=args.max_steps,
            min_amp_step=args.min_amp_step,
            rcond=args.rcond,
            tol=args.tol,
            kick=args.kick,
            escape_size=args.escape_size,
            escape_trials=args.escape_trials,
        )


if __name__ == "__main__":
    main()
