from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from geometry_flow import (
    SpectrumInfo,
    amp_index,
    naive_theta,
    pinv_solve,
    residual_vector,
    robust_svd,
    svd_info,
    unpack,
)
from rankflow import mm_tensor


torch.set_default_dtype(torch.float64)


def physical_constraints(theta: torch.Tensor, target: torch.Tensor, n: int, rank: int) -> torch.Tensor:
    """Exact tensor equality plus unit-norm CP directions.

    geometry_flow.py normalizes factor columns inside reconstruct().  The extra
    norm constraints here quotient the corresponding radial gauge explicitly,
    leaving only physical tangent directions.
    """
    U, V, W, _ = unpack(theta, n, rank)
    norms = torch.cat([0.5 * ((X * X).sum(dim=0) - 1.0) for X in (U, V, W)])
    return torch.cat([residual_vector(theta, target, n, rank), norms])


def constraint_jacobian(
    theta: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    *,
    center: torch.Tensor | None = None,
    radius: float | None = None,
) -> torch.Tensor:
    x = theta.detach().clone().requires_grad_(True)
    return torch.autograd.functional.jacobian(
        lambda q: constraint_vector(q, target, n, rank, center=center, radius=radius),
        x,
        vectorize=True,
    ).detach()


def constraint_vector(
    theta: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    *,
    center: torch.Tensor | None = None,
    radius: float | None = None,
) -> torch.Tensor:
    base = physical_constraints(theta, target, n, rank)
    if center is None or radius is None:
        return base
    shell = 0.5 * (((theta - center) @ (theta - center)) - radius * radius)
    return torch.cat([base, shell.reshape(1)])


def truncated_pinv(A: torch.Tensor, rcond: float) -> torch.Tensor:
    U, S, Vh = robust_svd(A, full_matrices=False)
    if len(S) == 0:
        return torch.zeros((A.shape[1], A.shape[0]), dtype=A.dtype, device=A.device)
    keep = S > rcond * float(S[0])
    if not bool(keep.any()):
        return torch.zeros((A.shape[1], A.shape[0]), dtype=A.dtype, device=A.device)
    return Vh[keep, :].T @ torch.diag(1.0 / S[keep]) @ U[:, keep].T


def tangent_basis(J: torch.Tensor, rcond: float) -> tuple[torch.Tensor, SpectrumInfo]:
    _, _, Vh, info = svd_info(J, rcond=rcond, full=True)
    return Vh[info.rank :, :].T, info


def correct_constraints(
    theta: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    *,
    center: torch.Tensor | None = None,
    radius: float | None = None,
    tol: float = 1.0e-11,
    max_iters: int = 30,
    rcond: float = 1.0e-10,
) -> tuple[torch.Tensor, bool, int, float, float]:
    x = theta.detach().clone()
    total_step = 0.0
    for iteration in range(max_iters + 1):
        r = constraint_vector(x, target, n, rank, center=center, radius=radius).detach()
        rn = float(r.norm())
        if rn <= tol:
            return x, True, iteration, rn, total_step
        if iteration == max_iters:
            break
        J = constraint_jacobian(x, target, n, rank, center=center, radius=radius)
        delta = pinv_solve(J, -r, rcond=rcond)
        dn = float(delta.norm())
        if dn == 0.0:
            break
        alpha = 1.0
        accepted = False
        for _ in range(12):
            candidate = x + alpha * delta
            cr = float(
                constraint_vector(candidate, target, n, rank, center=center, radius=radius).norm()
            )
            if cr < rn:
                x = candidate
                total_step += alpha * dn
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    rn = float(constraint_vector(x, target, n, rank, center=center, radius=radius).norm())
    return x, rn <= tol, max_iters, rn, total_step


def amplitude_curvature_operator(
    theta: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    channel: int,
    *,
    rcond: float,
):
    """Second fundamental form seen by one channel amplitude.

    Let G(theta)=0 be exact multiplication plus the unit-norm gauge fixing.
    For tangent q, a second-order exact path has normal acceleration z solving

        J z = -D^2 G[q,q].

    The minimum-norm normal acceleration gives

        d^2 a_r / ds^2 = q^T K_r q.

    K_r is returned in an orthonormal basis of ker(J).
    """
    J = constraint_jacobian(theta, target, n, rank)
    N, info = tangent_basis(J, rcond)
    Jplus = truncated_pinv(J, rcond)

    e = torch.zeros(theta.numel(), dtype=theta.dtype, device=theta.device)
    e[amp_index(n, rank, channel)] = 1.0
    lam = Jplus.T @ e

    x = theta.detach().clone().requires_grad_(True)
    Hphi = torch.autograd.functional.hessian(
        lambda q: torch.dot(lam, physical_constraints(q, target, n, rank)),
        x,
        vectorize=True,
    ).detach()

    K = -(N.T @ Hphi @ N)
    K = 0.5 * (K + K.T)
    evals, evecs = torch.linalg.eigh(K)
    return K, evals, evecs, N, Jplus, info


def constraint_second_directional(
    theta: torch.Tensor,
    q: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
) -> torch.Tensor:
    x = theta.detach().clone().requires_grad_(True)

    def first_directional(xx: torch.Tensor) -> torch.Tensor:
        _, jv = torch.autograd.functional.jvp(
            lambda yy: physical_constraints(yy, target, n, rank),
            xx,
            q,
            create_graph=True,
        )
        return jv

    _, hqq = torch.autograd.functional.jvp(first_directional, x, q, create_graph=False)
    return hqq.detach()


def physical_killability(
    theta: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    *,
    rcond: float,
):
    J = constraint_jacobian(theta, target, n, rank)
    N, info = tangent_basis(J, rcond)
    _, _, _, a = unpack(theta, n, rank)
    rows = []
    for ch in range(rank):
        e = torch.zeros(theta.numel(), dtype=theta.dtype, device=theta.device)
        e[amp_index(n, rank, ch)] = 1.0
        p = N @ (N.T @ e) if N.shape[1] else torch.zeros_like(e)
        score = float(p.norm())
        rows.append(
            {
                "channel": ch,
                "amplitude": float(a[ch]),
                "killability": score,
                "unit_change_cost": (1.0 / score) if score > 1.0e-14 else math.inf,
            }
        )
    return rows, info


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def curvature_report(*, n: int, out: Path, device: str, rcond: float):
    rank = n**3
    target = mm_tensor(n, device)
    theta = naive_theta(n, device)
    J = constraint_jacobian(theta, target, n, rank)
    N, info = tangent_basis(J, rcond)

    rows = []
    operators = []
    for ch in range(rank):
        K, evals, _, _, _, _ = amplitude_curvature_operator(
            theta, target, n, rank, ch, rcond=rcond
        )
        operators.append(K)
        positive = evals[evals > 1.0e-8]
        negative = evals[evals < -1.0e-8]
        rows.append(
            {
                "channel": ch,
                "tangent_dim": N.shape[1],
                "min_curvature": float(evals[0]),
                "max_curvature": float(evals[-1]),
                "positive_rank": int(len(positive)),
                "negative_rank": int(len(negative)),
                "positive_eigenvalues": ";".join(f"{float(v):.12g}" for v in positive),
            }
        )

    total = sum(operators)
    total_eigs = torch.linalg.eigvalsh(total)
    payload = {
        "n": n,
        "rank": rank,
        "constraint_jacobian": asdict(info),
        "physical_tangent_dimension": int(N.shape[1]),
        "channels": rows,
        "aggregate_curvature_eigenvalues": [float(v) for v in total_eigs],
    }
    write_csv(out / "curvature_channels.csv", rows)
    with (out / "curvature.json").open("w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"physical J={info.rows}x{info.cols}, rank={info.rank}, "
        f"tangent_dim={N.shape[1]}"
    )
    for row in rows:
        print(
            f"channel {row['channel']}: curvature=[{row['min_curvature']:.3e}, "
            f"{row['max_curvature']:.3e}], +rank={row['positive_rank']}, "
            f"-rank={row['negative_rank']}"
        )
    nz = total_eigs[total_eigs.abs() > 1.0e-8]
    print("aggregate nonzero curvature eigenvalues:", [float(v) for v in nz])


def curvature_escape(
    theta: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    *,
    channel: int,
    size: float,
    rcond: float,
    tol: float,
):
    K, evals, evecs, N, Jplus, _ = amplitude_curvature_operator(
        theta, target, n, rank, channel, rcond=rcond
    )
    # At the schoolbook point all amplitudes are positive and K is PSD.  The
    # top eigenvector creates the largest second-order amplitude motion and,
    # empirically, the largest first-order killability after a finite step.
    zeta = evecs[:, -1]
    q = N @ zeta
    q = q / q.norm()
    hqq = constraint_second_directional(theta, q, target, n, rank)
    acceleration = -Jplus @ hqq
    predictor = theta + size * q + 0.5 * size * size * acceleration
    corrected, ok, iters, residual, corrector_norm = correct_constraints(
        predictor,
        target,
        n,
        rank,
        tol=tol,
        max_iters=35,
        rcond=rcond,
    )
    kills, info = physical_killability(corrected, target, n, rank, rcond=rcond)
    _, _, _, a = unpack(corrected, n, rank)
    return corrected, {
        "channel": channel,
        "size": size,
        "predicted_curvature": float(evals[-1]),
        "converged": ok,
        "corrector_iters": iters,
        "constraint_residual": residual,
        "corrector_norm": corrector_norm,
        "distance": float((corrected - theta).norm()),
        "amplitude": float(a[channel]),
        "killability": float(kills[channel]["killability"]),
        "best_killability": max(float(row["killability"]) for row in kills),
        "sigma_min_positive": info.sigma_min_positive,
    }


def shell_minimize_amplitude(
    start: torch.Tensor,
    center: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    *,
    channel: int,
    step_size: float,
    max_steps: int,
    rcond: float,
    tol: float,
):
    radius = float((start - center).norm())
    theta, ok, _, residual, _ = correct_constraints(
        start,
        target,
        n,
        rank,
        center=center,
        radius=radius,
        tol=tol,
        max_iters=35,
        rcond=rcond,
    )
    if not ok:
        return theta, [], {
            "converged": False,
            "radius": radius,
            "final_constraint_residual": residual,
            "status": "initial_shell_correction_failed",
        }

    e = torch.zeros(theta.numel(), dtype=theta.dtype, device=theta.device)
    e[amp_index(n, rank, channel)] = 1.0
    rows = []
    status = "max_steps"

    for step in range(max_steps):
        J = constraint_jacobian(
            theta, target, n, rank, center=center, radius=radius
        )
        N, info = tangent_basis(J, rcond)
        p = N @ (N.T @ e) if N.shape[1] else torch.zeros_like(e)
        kill = float(p.norm())
        _, _, _, a = unpack(theta, n, rank)
        ak = float(a[channel])
        rows.append(
            {
                "step": step,
                "amplitude": ak,
                "shell_killability": kill,
                "radius": radius,
                "constraint_residual": float(
                    constraint_vector(
                        theta, target, n, rank, center=center, radius=radius
                    ).norm()
                ),
                "jacobian_rank": info.rank,
                "sigma_min_positive": info.sigma_min_positive,
            }
        )
        if kill < 1.0e-10:
            status = "shell_stationary"
            break

        direction = -p / kill if ak >= 0 else p / kill
        ds = step_size
        accepted = False
        while ds >= 1.0e-7:
            candidate, cok, _, _, _ = correct_constraints(
                theta + ds * direction,
                target,
                n,
                rank,
                center=center,
                radius=radius,
                tol=tol,
                max_iters=35,
                rcond=rcond,
            )
            if cok:
                ca = float(unpack(candidate, n, rank)[3][channel])
                if abs(ca) < abs(ak) - 1.0e-10:
                    theta = candidate
                    accepted = True
                    break
            ds *= 0.5
        if not accepted:
            status = "shell_descent_stalled"
            break

    _, _, _, a = unpack(theta, n, rank)
    summary = {
        "converged": True,
        "radius": radius,
        "channel": channel,
        "final_amplitude": float(a[channel]),
        "status": status,
        "steps": len(rows),
    }
    return theta, rows, summary


def profile(*, n: int, out: Path, device: str, rcond: float, tol: float, channel: int, sizes: list[float]):
    rank = n**3
    target = mm_tensor(n, device)
    theta0 = naive_theta(n, device)
    rows = []

    for size in sizes:
        escaped, erow = curvature_escape(
            theta0,
            target,
            n,
            rank,
            channel=channel,
            size=size,
            rcond=rcond,
            tol=tol,
        )
        minimized, shell_rows, srow = shell_minimize_amplitude(
            escaped,
            theta0,
            target,
            n,
            rank,
            channel=channel,
            step_size=0.04,
            max_steps=140,
            rcond=rcond,
            tol=tol,
        )
        write_csv(out / f"shell_size_{size:g}.csv", shell_rows)
        rows.append(
            {
                "escape_size": size,
                "escape_distance": erow["distance"],
                "escape_amplitude": erow["amplitude"],
                "escape_killability": erow["killability"],
                "shell_radius": srow["radius"],
                "shell_min_amplitude": srow.get("final_amplitude", math.nan),
                "shell_status": srow["status"],
            }
        )
        print(
            f"size={size:g}: escape a={erow['amplitude']:.6f}, "
            f"kill={erow['killability']:.3f}, shell min a={rows[-1]['shell_min_amplitude']:.9f}"
        )

    write_csv(out / "shell_profile.csv", rows)
    with (out / "shell_profile.json").open("w") as f:
        json.dump(rows, f, indent=2)


def parse_sizes(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Second-order geometry for matrix multiplication rank flow")
    p.add_argument("--mode", choices=["curvature", "escape", "profile", "all"], default="all")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("runs/curvature_flow"))
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--size", type=float, default=0.8)
    p.add_argument("--sizes", default="0.8,1.2,1.6")
    p.add_argument("--rcond", type=float, default=1.0e-10)
    p.add_argument("--tol", type=float, default=1.0e-11)
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rank = args.n**3
    target = mm_tensor(args.n, args.device)
    theta0 = naive_theta(args.n, args.device)

    if args.mode in ("curvature", "all"):
        curvature_report(n=args.n, out=args.out, device=args.device, rcond=args.rcond)

    if args.mode in ("escape", "all"):
        escaped, row = curvature_escape(
            theta0,
            target,
            args.n,
            rank,
            channel=args.channel,
            size=args.size,
            rcond=args.rcond,
            tol=args.tol,
        )
        torch.save(escaped.cpu(), args.out / "curvature_escape_theta.pt")
        with (args.out / "curvature_escape.json").open("w") as f:
            json.dump(row, f, indent=2)
        print(json.dumps(row, indent=2))

    if args.mode in ("profile", "all"):
        profile(
            n=args.n,
            out=args.out,
            device=args.device,
            rcond=args.rcond,
            tol=args.tol,
            channel=args.channel,
            sizes=parse_sizes(args.sizes),
        )


if __name__ == "__main__":
    main()
