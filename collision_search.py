from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch

from curvature_flow import (
    amplitude_curvature_operator,
    constraint_jacobian,
    constraint_second_directional,
    correct_constraints,
    physical_constraints,
    tangent_basis,
    truncated_pinv,
)
from geometry_flow import naive_theta, pack, pinv_solve, reconstruct, unpack
from rankflow import mm_tensor


torch.set_default_dtype(torch.float64)

POSITIVE_PRODUCT_SIGNS = [
    tuple(float(v) for v in signs)
    for signs in itertools.product((-1, 1), repeat=3)
    if math.prod(signs) == 1
]


def collision_value(
    theta: torch.Tensor,
    n: int,
    rank: int,
    pair: tuple[int, int],
    signs: tuple[float, float, float],
) -> torch.Tensor:
    """Mean signed factor cosine for one channel pair.

    Unit norms are included in physical_constraints(), so on the exact physical
    manifold this lies in [-1,1].  A value +1 with prod(signs)=+1 means the two
    rank-one tensors are identical and may be fused by adding amplitudes.
    """
    r, s = pair
    U, V, W, _ = unpack(theta, n, rank)
    return sum(
        sign * torch.dot(X[:, r], X[:, s])
        for sign, X in zip(signs, (U, V, W))
    ) / 3.0


def factor_cosines(
    theta: torch.Tensor,
    n: int,
    rank: int,
    pair: tuple[int, int],
) -> tuple[float, float, float]:
    r, s = pair
    U, V, W, _ = unpack(theta, n, rank)
    return tuple(float(torch.dot(X[:, r], X[:, s])) for X in (U, V, W))


def constrained_objective_hessian(
    theta: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    N: torch.Tensor,
    Jplus: torch.Tensor,
    pair: tuple[int, int],
    signs: tuple[float, float, float],
) -> tuple[torch.Tensor, float]:
    """Riemannian/constrained Hessian of collision_value on G(theta)=0.

    At a constrained stationary point grad f = J^T lambda.  Along an exact
    second-order path the curvature is

        q^T [H_f - sum_i lambda_i H_{G_i}] q.

    Returning N^T (...) N avoids the false ambient-Hessian signal that a plain
    second derivative gives at schoolbook.
    """
    x = theta.detach().clone().requires_grad_(True)
    f = collision_value(x, n, rank, pair, signs)
    g = torch.autograd.grad(f, x)[0].detach()
    tangent_grad_norm = float((N.T @ g).norm()) if N.shape[1] else 0.0
    lam = Jplus.T @ g

    Hf = torch.autograd.functional.hessian(
        lambda q: collision_value(q, n, rank, pair, signs),
        x,
        vectorize=True,
    ).detach()
    HG = torch.autograd.functional.hessian(
        lambda q: torch.dot(lam, physical_constraints(q, target, n, rank)),
        x,
        vectorize=True,
    ).detach()
    K = N.T @ (Hf - HG) @ N
    K = 0.5 * (K + K.T)
    return K, tangent_grad_norm


def scan_schoolbook(
    *,
    n: int = 2,
    device: str = "cpu",
    rcond: float = 1.0e-10,
    amplitude_curvature_tol: float = 1.0e-8,
) -> tuple[list[dict], dict[tuple[int, int], torch.Tensor]]:
    """Score every channel pair without any Strassen pair being specified.

    Useful collision directions must satisfy three local requirements:

    1. constructive sign product (+1), so a collision can fuse rather than
       cancel;
    2. no first-order collision motion at schoolbook (we are looking for the
       hidden second-order seam, not pairs already sharing a factor);
    3. zero second-order growth of the *two colliding amplitudes*.

    The remaining signal is the constrained second-order collision curvature.
    """
    rank = n**3
    target = mm_tensor(n, device)
    theta = naive_theta(n, device)
    J = constraint_jacobian(theta, target, n, rank)
    N, info = tangent_basis(J, rcond)
    Jplus = truncated_pinv(J, rcond)

    amp_ops = [
        amplitude_curvature_operator(theta, target, n, rank, ch, rcond=rcond)[0]
        for ch in range(rank)
    ]

    rows: list[dict] = []
    directions: dict[tuple[int, int], torch.Tensor] = {}

    for pair in itertools.combinations(range(rank), 2):
        r, s = pair
        pair_amp = 0.5 * (amp_ops[r] + amp_ops[r].T) + 0.5 * (amp_ops[s] + amp_ops[s].T)
        aevals, aevecs = torch.linalg.eigh(pair_amp)
        zero_amp_basis = aevecs[:, aevals < amplitude_curvature_tol]

        best = None
        for signs in POSITIVE_PRODUCT_SIGNS:
            K, tangent_grad = constrained_objective_hessian(
                theta, target, n, rank, N, Jplus, pair, signs
            )
            if zero_amp_basis.shape[1]:
                Kr = zero_amp_basis.T @ K @ zero_amp_basis
                evals, evecs = torch.linalg.eigh(0.5 * (Kr + Kr.T))
                curvature = float(evals[-1])
                zeta = zero_amp_basis @ evecs[:, -1]
                q = N @ zeta
                q = q / q.norm().clamp_min(1.0e-30)
            else:
                curvature = -math.inf
                q = torch.zeros(theta.numel(), dtype=theta.dtype, device=theta.device)

            item = {
                "pair": pair,
                "signs": signs,
                "curvature": curvature,
                "tangent_grad_norm": tangent_grad,
                "zero_pair_amp_dim": int(zero_amp_basis.shape[1]),
                "direction": q,
            }
            if best is None or item["curvature"] > best["curvature"]:
                best = item

        assert best is not None
        cos = factor_cosines(theta, n, rank, pair)
        row = {
            "r": r,
            "s": s,
            "initial_cos_u": cos[0],
            "initial_cos_v": cos[1],
            "initial_cos_w": cos[2],
            "best_sign_u": best["signs"][0],
            "best_sign_v": best["signs"][1],
            "best_sign_w": best["signs"][2],
            "tangent_grad_norm": best["tangent_grad_norm"],
            "zero_pair_amp_dim": best["zero_pair_amp_dim"],
            "constrained_collision_curvature": best["curvature"],
        }
        rows.append(row)
        directions[pair] = best["direction"].detach()

    rows.sort(key=lambda x: (-x["constrained_collision_curvature"], x["r"], x["s"]))
    meta = {
        "jacobian": asdict(info),
        "physical_tangent_dim": int(N.shape[1]),
    }
    return rows, directions


def choose_hidden_pair(
    rows: list[dict],
    *,
    seed: int = 0,
    first_order_tol: float = 1.0e-8,
    tie_tol: float = 1.0e-10,
) -> tuple[dict, list[dict]]:
    hidden = [r for r in rows if r["tangent_grad_norm"] < first_order_tol]
    if not hidden:
        raise RuntimeError("no second-order-only collision candidates found")
    best_curv = max(r["constrained_collision_curvature"] for r in hidden)
    tied = [r for r in hidden if best_curv - r["constrained_collision_curvature"] <= tie_tol]
    rng = random.Random(seed)
    chosen = tied[rng.randrange(len(tied))]
    return chosen, tied


def second_order_escape(
    theta: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    q: torch.Tensor,
    *,
    size: float,
    rcond: float,
    tol: float,
) -> torch.Tensor:
    J = constraint_jacobian(theta, target, n, rank)
    Jplus = truncated_pinv(J, rcond)
    hqq = constraint_second_directional(theta, q, target, n, rank)
    acceleration = -Jplus @ hqq
    predictor = theta + size * q + 0.5 * size * size * acceleration
    corrected, ok, _, residual, _ = correct_constraints(
        predictor,
        target,
        n,
        rank,
        tol=tol,
        max_iters=50,
        rcond=rcond,
    )
    if not ok:
        raise RuntimeError(f"second-order escape correction failed: residual={residual:.3e}")
    return corrected


def exact_collision_correct(
    theta: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    pair: tuple[int, int],
    signs: tuple[float, float, float],
    *,
    tol: float = 1.0e-12,
    rcond: float = 1.0e-10,
    max_iters: int = 30,
) -> tuple[torch.Tensor, bool, float]:
    r, s = pair

    def constraints(q: torch.Tensor) -> torch.Tensor:
        U, V, W, _ = unpack(q, n, rank)
        diffs = torch.cat(
            [
                U[:, r] - signs[0] * U[:, s],
                V[:, r] - signs[1] * V[:, s],
                W[:, r] - signs[2] * W[:, s],
            ]
        )
        return torch.cat([physical_constraints(q, target, n, rank), diffs])

    x = theta.detach().clone()
    for _ in range(max_iters):
        residual = constraints(x).detach()
        rn = float(residual.norm())
        if rn <= tol:
            return x, True, rn
        xx = x.detach().clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(constraints, xx, vectorize=True).detach()
        delta = pinv_solve(J, -residual, rcond=rcond)
        alpha = 1.0
        accepted = False
        for _ in range(16):
            candidate = x + alpha * delta
            if float(constraints(candidate).norm()) < rn:
                x = candidate
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    rn = float(constraints(x).norm())
    return x, rn <= tol, rn


def fuse_pair(
    theta: torch.Tensor,
    n: int,
    rank: int,
    pair: tuple[int, int],
    signs: tuple[float, float, float],
):
    if abs(math.prod(signs) - 1.0) > 1.0e-12:
        raise ValueError("only constructive (+1 product) collisions can be fused by amplitude addition")
    r, s = pair
    U, V, W, a = unpack(theta, n, rank)
    keep = [ch for ch in range(rank) if ch != s]
    Uf, Vf, Wf, af = U[:, keep].clone(), V[:, keep].clone(), W[:, keep].clone(), a[keep].clone()
    rr = keep.index(r)
    af[rr] = a[r] + a[s]
    return Uf, Vf, Wf, af


def collision_ascent(
    *,
    n: int = 2,
    device: str = "cpu",
    seed: int = 0,
    escape_size: float = 0.8,
    step_size: float = 0.08,
    amplitude_penalty: float = 0.01,
    max_steps: int = 100,
    collision_threshold: float = 0.99999,
    rcond: float = 1.0e-10,
    tol: float = 1.0e-10,
    out: Path | None = None,
) -> dict:
    rank = n**3
    target = mm_tensor(n, device)
    theta0 = naive_theta(n, device)

    rows, directions = scan_schoolbook(n=n, device=device, rcond=rcond)
    chosen, tied = choose_hidden_pair(rows, seed=seed)
    pair = (int(chosen["r"]), int(chosen["s"]))
    signs = (
        float(chosen["best_sign_u"]),
        float(chosen["best_sign_v"]),
        float(chosen["best_sign_w"]),
    )
    q = directions[pair]

    theta = second_order_escape(
        theta0, target, n, rank, q, size=escape_size, rcond=rcond, tol=tol
    )

    trajectory = []
    status = "max_steps"
    r, s = pair

    for step in range(max_steps):
        U, V, W, a = unpack(theta, n, rank)
        value = float(collision_value(theta, n, rank, pair, signs))
        cos = factor_cosines(theta, n, rank, pair)
        trajectory.append(
            {
                "step": step,
                "collision_value": value,
                "cos_u": cos[0],
                "cos_v": cos[1],
                "cos_w": cos[2],
                "amp_r": float(a[r]),
                "amp_s": float(a[s]),
                "max_abs_amplitude": float(a.abs().max()),
                "constraint_residual": float(physical_constraints(theta, target, n, rank).norm()),
            }
        )
        if value >= collision_threshold:
            status = "collision_threshold"
            break

        x = theta.detach().clone().requires_grad_(True)
        _, _, _, ax = unpack(x, n, rank)
        objective = collision_value(x, n, rank, pair, signs) - amplitude_penalty * (
            (ax[r] - 1.0) ** 2 + (ax[s] - 1.0) ** 2
        )
        g = torch.autograd.grad(objective, x)[0].detach()
        J = constraint_jacobian(theta, target, n, rank)
        N, _ = tangent_basis(J, rcond)
        pg = N @ (N.T @ g)
        pg_norm = float(pg.norm())
        if pg_norm < 1.0e-12:
            status = "projected_gradient_stalled"
            break
        direction = pg / pg_norm
        old_objective = float(objective.detach())

        ds = step_size
        accepted = False
        for _ in range(14):
            candidate, ok, _, _, _ = correct_constraints(
                theta + ds * direction,
                target,
                n,
                rank,
                tol=tol,
                max_iters=35,
                rcond=rcond,
            )
            if ok:
                _, _, _, ac = unpack(candidate, n, rank)
                candidate_objective = float(
                    collision_value(candidate, n, rank, pair, signs)
                    - amplitude_penalty * ((ac[r] - 1.0) ** 2 + (ac[s] - 1.0) ** 2)
                )
                if candidate_objective > old_objective + 1.0e-10:
                    theta = candidate
                    accepted = True
                    break
            ds *= 0.5
        if not accepted:
            status = "line_search_stalled"
            break

    pre_force_value = float(collision_value(theta, n, rank, pair, signs))
    exact_theta, collision_ok, collision_residual = exact_collision_correct(
        theta, target, n, rank, pair, signs, tol=1.0e-12, rcond=rcond
    )
    Uf, Vf, Wf, af = fuse_pair(exact_theta, n, rank, pair, signs)
    fused_tensor = torch.einsum("ir,jr,kr,r->ijk", Uf, Vf, Wf, af)
    fused_residual = float((fused_tensor - target).norm())

    _, _, _, exact_a = unpack(exact_theta, n, rank)
    summary = {
        "n": n,
        "starting_rank": rank,
        "seed": seed,
        "tied_best_pairs": [[int(rw["r"]), int(rw["s"])] for rw in tied],
        "chosen_pair": list(pair),
        "chosen_signs": list(signs),
        "schoolbook_collision_curvature": chosen["constrained_collision_curvature"],
        "schoolbook_tangent_gradient_norm": chosen["tangent_grad_norm"],
        "pre_force_collision_value": pre_force_value,
        "collision_corrector_ok": collision_ok,
        "collision_constraint_residual": collision_residual,
        "exact_pair_amplitudes": [float(exact_a[r]), float(exact_a[s])],
        "fused_rank": rank - 1,
        "fused_tensor_residual": fused_residual,
        "status": status,
        "trajectory_steps": len(trajectory),
    }

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "pair_scan.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (out / "trajectory.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(trajectory[0].keys()))
            writer.writeheader()
            writer.writerows(trajectory)
        with (out / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        torch.save(theta.cpu(), out / "pre_force_theta.pt")
        torch.save(exact_theta.cpu(), out / "exact_collision_theta.pt")
        torch.save(
            {"U": Uf.cpu(), "V": Vf.cpu(), "W": Wf.cpu(), "a": af.cpu()},
            out / "rank7_fused.pt",
        )

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Autonomous collision search from schoolbook matmul")
    p.add_argument("--mode", choices=["scan", "search"], default="search")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--escape-size", type=float, default=0.8)
    p.add_argument("--step-size", type=float, default=0.08)
    p.add_argument("--amplitude-penalty", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("runs/collision_search"))
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "scan":
        rows, _ = scan_schoolbook(n=args.n, device=args.device)
        with (args.out / "pair_scan.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            print(
                f"({row['r']},{row['s']}): curvature={row['constrained_collision_curvature']:.12g}, "
                f"|grad_T|={row['tangent_grad_norm']:.3e}, "
                f"zero_amp_dim={row['zero_pair_amp_dim']}"
            )
        return

    summary = collision_ascent(
        n=args.n,
        device=args.device,
        seed=args.seed,
        escape_size=args.escape_size,
        step_size=args.step_size,
        amplitude_penalty=args.amplitude_penalty,
        max_steps=args.max_steps,
        out=args.out,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
