from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from pathlib import Path

import torch

from analytic_ansatz import SQRT2, rank7_family
from collision_search import (
    POSITIVE_PRODUCT_SIGNS,
    constrained_objective_hessian,
)
from curvature_flow import (
    amplitude_curvature_operator,
    constraint_jacobian,
    tangent_basis,
    truncated_pinv,
    physical_constraints,
)
from geometry_flow import naive_theta, pack, unpack
from rankflow import mm_tensor, naive_factors


torch.set_default_dtype(torch.float64)


def channel_to_ijk(ch: int, n: int) -> tuple[int, int, int]:
    i, rem = divmod(ch, n * n)
    j, k = divmod(rem, n)
    return i, j, k


def ijk_to_channel(i: int, j: int, k: int, n: int) -> int:
    return i * n * n + j * n + k


def pair_mask(pair: tuple[int, int], n: int) -> tuple[int, int, int]:
    a = channel_to_ijk(pair[0], n)
    b = channel_to_ijk(pair[1], n)
    return tuple(int(x != y) for x, y in zip(a, b))


def mask_name(mask: tuple[int, int, int]) -> str:
    return "".join(str(x) for x in mask)


def representative_channel(mask: tuple[int, int, int], n: int) -> int:
    # Schoolbook symmetry lets us represent every orbit by (0,0,0) and a
    # channel whose differing coordinates are 1.
    return ijk_to_channel(mask[0], mask[1], mask[2], n)


def orbit_counts(n: int) -> dict[tuple[int, int, int], int]:
    counts: dict[tuple[int, int, int], int] = {}
    rank = n**3
    for pair in itertools.combinations(range(rank), 2):
        m = pair_mask(pair, n)
        counts[m] = counts.get(m, 0) + 1
    return counts


def schoolbook_orbit_scan(
    *,
    n: int = 3,
    device: str = "cpu",
    rcond: float = 1.0e-10,
    amplitude_curvature_tol: float = 1.0e-8,
) -> tuple[list[dict], dict]:
    """Exact constrained-curvature scan using schoolbook symmetry orbits.

    Instead of evaluating all C(n^3, 2) pairs, evaluate the 2^3-1 equality
    masks between channel triples (i,j,k).  Independent relabellings of i,j,k
    make all pairs with the same mask equivalent at schoolbook.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    rank = n**3
    target = mm_tensor(n, device)
    theta = naive_theta(n, device)
    J = constraint_jacobian(theta, target, n, rank)
    N, info = tangent_basis(J, rcond)
    Jplus = truncated_pinv(J, rcond)

    masks = [m for m in itertools.product((0, 1), repeat=3) if any(m)]
    rep_channels = {representative_channel(m, n) for m in masks} | {0}
    amp_ops = {
        ch: amplitude_curvature_operator(theta, target, n, rank, ch, rcond=rcond)[0]
        for ch in sorted(rep_channels)
    }
    counts = orbit_counts(n)

    rows: list[dict] = []
    for mask in masks:
        ch = representative_channel(mask, n)
        pair = (0, ch)
        pair_amp = 0.5 * (amp_ops[0] + amp_ops[0].T) + 0.5 * (
            amp_ops[ch] + amp_ops[ch].T
        )
        aevals, aevecs = torch.linalg.eigh(pair_amp)
        zero_amp_basis = aevecs[:, aevals < amplitude_curvature_tol]

        best = None
        for signs in POSITIVE_PRODUCT_SIGNS:
            K, tangent_grad = constrained_objective_hessian(
                theta, target, n, rank, N, Jplus, pair, signs
            )
            Kr = zero_amp_basis.T @ K @ zero_amp_basis
            evals, evecs = torch.linalg.eigh(0.5 * (Kr + Kr.T))
            curvature = float(evals[-1])
            item = (curvature, signs, tangent_grad)
            if best is None or curvature > best[0]:
                best = item

        assert best is not None
        rows.append(
            {
                "mask": mask_name(mask),
                "hamming_distance": sum(mask),
                "pair_count": counts[mask],
                "representative_r": 0,
                "representative_s": ch,
                "tangent_gradient_norm": float(best[2]),
                "zero_pair_amp_dim": int(zero_amp_basis.shape[1]),
                "constrained_collision_curvature": float(best[0]),
                "best_sign_u": float(best[1][0]),
                "best_sign_v": float(best[1][1]),
                "best_sign_w": float(best[1][2]),
            }
        )

    rows.sort(key=lambda x: (-x["constrained_collision_curvature"], x["mask"]))
    meta = {
        "n": n,
        "rank": rank,
        "total_pairs": math.comb(rank, 2),
        "physical_jacobian_rows": info.rows,
        "physical_jacobian_cols": info.cols,
        "physical_jacobian_rank": info.rank,
        "physical_tangent_dimension": int(N.shape[1]),
    }
    return rows, meta


def all_best_pairs(n: int, orbit_rows: list[dict], tol: float = 1e-10) -> list[tuple[int, int]]:
    best = orbit_rows[0]["constrained_collision_curvature"]
    masks = {
        tuple(int(c) for c in row["mask"])
        for row in orbit_rows
        if best - row["constrained_collision_curvature"] <= tol
        and row["tangent_gradient_norm"] < 1e-8
    }
    return [
        pair
        for pair in itertools.combinations(range(n**3), 2)
        if pair_mask(pair, n) in masks
    ]


def cube_from_opposite_pair(pair: tuple[int, int], n: int):
    a = channel_to_ijk(pair[0], n)
    b = channel_to_ijk(pair[1], n)
    if not all(x != y for x, y in zip(a, b)):
        raise ValueError("pair must differ in all three schoolbook indices")
    # Keep orientation: local 000 is pair[0], local 111 is pair[1].
    return (a[0], b[0]), (a[1], b[1]), (a[2], b[2])


def cube_channels(I, J, K, n: int) -> list[int]:
    return [ijk_to_channel(i, j, k, n) for i in I for j in J for k in K]


def scatter_local_factor(local: torch.Tensor, left, right, n: int, kind: str) -> torch.Tensor:
    """Scatter a 4-vector [00,01,10,11] into one global n^2 factor space."""
    out = torch.zeros(n * n, dtype=local.dtype, device=local.device)
    for x in range(2):
        for y in range(2):
            if kind == "U":
                idx = left[x] * n + right[y]
            elif kind == "V":
                idx = left[x] * n + right[y]
            elif kind == "W":
                idx = left[x] * n + right[y]
            else:
                raise ValueError(kind)
            out[idx] = local[x * 2 + y]
    return out


def embed_local_factors(
    U2: torch.Tensor,
    V2: torch.Tensor,
    W2: torch.Tensor,
    a2: torch.Tensor,
    *,
    I,
    J,
    K,
    n: int,
):
    R = a2.numel()
    U = torch.stack([scatter_local_factor(U2[:, r], I, J, n, "U") for r in range(R)], dim=1)
    V = torch.stack([scatter_local_factor(V2[:, r], J, K, n, "V") for r in range(R)], dim=1)
    W = torch.stack([scatter_local_factor(W2[:, r], I, K, n, "W") for r in range(R)], dim=1)
    return U, V, W, a2.clone()


def fuse_cube_to_rank_minus_one(
    pair: tuple[int, int],
    *,
    n: int = 3,
    device: str = "cpu",
):
    """Replace the 8 schoolbook products in the selected 2x2x2 cube by 7.

    The local 2x2 collision operator was discovered in the previous stage.
    Here the 3x3 curvature scan decides *where* that primitive should act.
    """
    I, J, K = cube_from_opposite_pair(pair, n)
    cube = set(cube_channels(I, J, K, n))

    U0, V0, W0 = naive_factors(n, device)
    keep = [r for r in range(n**3) if r not in cube]
    U_keep, V_keep, W_keep = U0[:, keep], V0[:, keep], W0[:, keep]
    a_keep = torch.ones(len(keep), dtype=torch.float64, device=device)

    # Classical Strassen + one zero duplicate in the orientation where local
    # channels 000 and 111 are the collision pair. Drop the zero channel.
    U8, V8, W8, a8 = rank7_family(0.0, 2.0 * SQRT2, device=device)
    live = [r for r in range(8) if abs(float(a8[r])) > 1e-12]
    U2, V2, W2, a2 = U8[:, live], V8[:, live], W8[:, live], a8[live]
    Ug, Vg, Wg, ag = embed_local_factors(U2, V2, W2, a2, I=I, J=J, K=K, n=n)

    U = torch.cat([U_keep, Ug], dim=1)
    V = torch.cat([V_keep, Vg], dim=1)
    W = torch.cat([W_keep, Wg], dim=1)
    a = torch.cat([a_keep, ag])
    target = mm_tensor(n, device)
    recon = torch.einsum("ir,jr,kr,r->ijk", U, V, W, a)
    return U, V, W, a, float((recon - target).norm()), {
        "I": list(I),
        "J": list(J),
        "K": list(K),
        "removed_schoolbook_channels": sorted(cube),
        "kept_schoolbook_channels": keep,
    }


def full_3x3_homotopy_sample(
    pair: tuple[int, int],
    *,
    images: int = 21,
    n: int = 3,
    device: str = "cpu",
):
    """Embed the exact 2x2 schoolbook->collision->fusion homotopy in 3x3."""
    from closed_form_homotopy import full_homotopy

    I, J, K = cube_from_opposite_pair(pair, n)
    cube = set(cube_channels(I, J, K, n))
    U0, V0, W0 = naive_factors(n, device)
    keep = [r for r in range(n**3) if r not in cube]
    target = mm_tensor(n, device)
    rows = []
    points = []

    for image, tt in enumerate(torch.linspace(0.0, 1.0, images, dtype=torch.float64)):
        t = float(tt)
        local_theta = full_homotopy(t, device=device)
        U2, V2, W2, a2 = unpack(local_theta, 2, 8)
        Ug, Vg, Wg, ag = embed_local_factors(U2, V2, W2, a2, I=I, J=J, K=K, n=n)
        U = torch.cat([U0[:, keep], Ug], dim=1)
        V = torch.cat([V0[:, keep], Vg], dim=1)
        W = torch.cat([W0[:, keep], Wg], dim=1)
        a = torch.cat([torch.ones(len(keep), device=device), ag])
        recon = torch.einsum("ir,jr,kr,r->ijk", U, V, W, a)
        residual = float((recon - target).norm())
        points.append(pack(U, V, W, a))
        rows.append(
            {
                "image": image,
                "t": t,
                "residual": residual,
                "rank_channels": int(a.numel()),
                "active_at_1e_10": int((a.abs() > 1e-10).sum()),
                "local_a0": float(ag[0]),
                "local_a7": float(ag[7]),
            }
        )
    return points, rows


def projected_pair_mobility(
    theta: torch.Tensor,
    target: torch.Tensor,
    n: int,
    rank: int,
    *,
    rcond: float = 1e-10,
) -> list[dict]:
    """Cheap first-order scan after symmetry has been broken by one fusion."""
    J = constraint_jacobian(theta, target, n, rank)
    N, info = tangent_basis(J, rcond)
    U, V, W, a = unpack(theta, n, rank)
    block = n * n * rank

    rows = []
    for r, s in itertools.combinations(range(rank), 2):
        best = None
        for signs in POSITIVE_PRODUCT_SIGNS:
            g = torch.zeros_like(theta)
            for which, (X, sign) in enumerate(zip((U, V, W), signs)):
                base = which * block
                # X is row-major (d, rank), so flat index = row*rank+channel.
                for row in range(n * n):
                    g[base + row * rank + r] += sign * X[row, s] / 3.0
                    g[base + row * rank + s] += sign * X[row, r] / 3.0
            pg_norm = float((N.T @ g).norm()) if N.shape[1] else 0.0
            value = sum(
                sign * float(torch.dot(X[:, r], X[:, s]))
                for sign, X in zip(signs, (U, V, W))
            ) / 3.0
            item = (pg_norm, value, signs)
            if best is None or pg_norm > best[0]:
                best = item
        rows.append(
            {
                "r": r,
                "s": s,
                "projected_gradient_norm": best[0],
                "collision_value": best[1],
                "sign_u": best[2][0],
                "sign_v": best[2][1],
                "sign_w": best[2][2],
                "amp_r": float(a[r]),
                "amp_s": float(a[s]),
            }
        )
    rows.sort(key=lambda x: (-x["projected_gradient_norm"], -x["collision_value"]))
    return rows


def run_demo(*, seed: int, out: Path, device: str = "cpu") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    orbit_rows, meta = schoolbook_orbit_scan(n=3, device=device)
    best_pairs = all_best_pairs(3, orbit_rows)
    chosen = random.Random(seed).choice(best_pairs)

    with (out / "orbit_scan.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(orbit_rows[0].keys()))
        writer.writeheader(); writer.writerows(orbit_rows)

    points, path_rows = full_3x3_homotopy_sample(chosen, images=31, n=3, device=device)
    with (out / "embedded_homotopy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(path_rows[0].keys()))
        writer.writeheader(); writer.writerows(path_rows)

    U, V, W, a, fused_residual, cube_meta = fuse_cube_to_rank_minus_one(chosen, n=3, device=device)
    torch.save({"U": U.cpu(), "V": V.cpu(), "W": W.cpu(), "a": a.cpu()}, out / "rank26.pt")

    theta26 = pack(U, V, W, a)
    mobility = projected_pair_mobility(theta26, mm_tensor(3, device), 3, 26)
    with (out / "rank26_first_order_pair_mobility.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mobility[0].keys()))
        writer.writeheader(); writer.writerows(mobility)

    summary = {
        **meta,
        "best_orbit_curvature": orbit_rows[0]["constrained_collision_curvature"],
        "best_orbit_masks": [
            row["mask"] for row in orbit_rows
            if orbit_rows[0]["constrained_collision_curvature"] - row["constrained_collision_curvature"] < 1e-10
        ],
        "best_pair_count": len(best_pairs),
        "chosen_pair": list(chosen),
        "chosen_pair_ijk": [list(channel_to_ijk(ch, 3)) for ch in chosen],
        "cube": cube_meta,
        "max_embedded_homotopy_residual": max(row["residual"] for row in path_rows),
        "endpoint_active_channels": path_rows[-1]["active_at_1e_10"],
        "fused_rank": int(a.numel()),
        "fused_tensor_residual": fused_residual,
        "rank26_top_first_order_pairs": mobility[:12],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="3x3 schoolbook collision-geometry experiment")
    p.add_argument("--mode", choices=["scan", "demo"], default="demo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("runs/collision_3x3"))
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "scan":
        rows, meta = schoolbook_orbit_scan(n=3, device=args.device)
        print(json.dumps(meta, indent=2))
        for row in rows:
            print(row)
        return
    print(json.dumps(run_demo(seed=args.seed, out=args.out, device=args.device), indent=2))


if __name__ == "__main__":
    main()
