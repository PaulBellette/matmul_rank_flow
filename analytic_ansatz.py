from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from geometry_flow import pack
from rankflow import mm_tensor


torch.set_default_dtype(torch.float64)

SQRT2 = math.sqrt(2.0)

# In the numerical string these are exact symmetry relations, not a fit:
# V[:, j] = U[:, INV_V[j]]
# W[:, j] = transpose(U[:, INV_W[j]]) in vec([11,12,21,22]) coordinates.
INV_V = (0, 2, 4, 6, 1, 3, 5, 7)
INV_W = (0, 4, 1, 5, 2, 6, 3, 7)
TRANSPOSE_ROWS = (0, 2, 1, 3)


def _as_tensor(values, *, device="cpu") -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float64, device=device)


def cyclic_factors_from_u(U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate V and W from one 4x8 factor matrix by the discovered symmetry."""
    V = U[:, list(INV_V)]
    W = U[list(TRANSPOSE_ROWS), :][:, list(INV_W)]
    return V, W


def branch_a_u(q: torch.Tensor) -> torch.Tensor:
    """The 10-coordinate schoolbook-connected symmetry sector.

    q = [A,B,C,D,E,F,G,H,I,J,x,y].  The corresponding factor matrix is

      [ A  B  C  D -C -D  E  F ]
      [ G  H  I  I  J  J -H -G ]
      [-G -H  J  J  I  I  H  G ]
      [ F  E -D -C  D  C  B  A ]

    with amplitudes [x,y,y,y,y,y,y,x].
    """
    A, B, C, D, E, F, G, H, I, J, _, _ = q
    return torch.stack(
        [
            torch.stack((A, B, C, D, -C, -D, E, F)),
            torch.stack((G, H, I, I, J, J, -H, -G)),
            torch.stack((-G, -H, J, J, I, I, H, G)),
            torch.stack((F, E, -D, -C, D, C, B, A)),
        ]
    )


def branch_a_amplitudes(q: torch.Tensor) -> torch.Tensor:
    x, y = q[10], q[11]
    return torch.stack((x, y, y, y, y, y, y, x))


def branch_a_tensor(q: torch.Tensor) -> torch.Tensor:
    U = branch_a_u(q)
    V, W = cyclic_factors_from_u(U)
    a = branch_a_amplitudes(q)
    return torch.einsum("ir,jr,kr,r->ijk", U, V, W, a)


def branch_a_full_residual(q: torch.Tensor) -> torch.Tensor:
    return (branch_a_tensor(q) - mm_tensor(2, str(q.device))).reshape(-1)


def branch_a_constraints(q: torch.Tensor) -> torch.Tensor:
    """Nine polynomial equations defining the exact physical symmetry sector.

    The 64 tensor entries collapse to only six distinct cubic equations under
    the symmetry above.  Three quadratic equations fix the unit-column gauge.
    Thus the exact set is generically 12 - 9 = 3 dimensional.
    """
    A, B, C, D, E, F, G, H, I, J, x, y = q

    e0 = A**3 * x - 3 * B * C**2 * y - 3 * D**2 * E * y + F**3 * x - 1
    e1 = (
        -A**2 * G * x
        + B * C * I * y
        - B * C * J * y
        + C**2 * H * y
        - D**2 * H * y
        + D * E * I * y
        - D * E * J * y
        + F**2 * G * x
    )
    e2 = A**2 * F * x + A * F**2 * x + 2 * B * C * D * y - B * D**2 * y - C**2 * E * y + 2 * C * D * E * y
    e3 = -A * G**2 * x + B * I**2 * y - 2 * C * H * J * y - 2 * D * H * I * y + E * J**2 * y - F * G**2 * x - 1
    e4 = A * G**2 * x + B * I * J * y - C * H * I * y + C * H * J * y + D * H * I * y - D * H * J * y + E * I * J * y + F * G**2 * x
    e5 = -A * G**2 * x + B * J**2 * y + 2 * C * H * I * y + 2 * D * H * J * y + E * I**2 * y - F * G**2 * x

    # Only three distinct column norms occur in this symmetry sector.
    n0 = A**2 + 2 * G**2 + F**2 - 1
    n1 = B**2 + 2 * H**2 + E**2 - 1
    n2 = C**2 + D**2 + I**2 + J**2 - 1
    return torch.stack((e0, e1, e2, e3, e4, e5, n0, n1, n2))


def schoolbook_reduced(*, device="cpu") -> torch.Tensor:
    return _as_tensor((1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1), device=device)


def rank7_family(theta: float, split: float, *, device="cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form exact rank-7 family with a duplicated channel.

    Channels 0 and 7 are identical rank-one tensors.  Their amplitudes may be
    split arbitrarily as p and 2*sqrt(2)-p without changing the represented
    multiplication tensor.

    theta=0 and split=2*sqrt(2) is classical Strassen (+ one zero channel).
    theta=0 and split=sqrt(2) is the equal-split fusion point used by the
    schoolbook-to-fusion homotopy.
    """
    s = math.sin(theta)
    c = math.cos(theta)
    denom = (1.0 + math.sin(2.0 * theta)) * math.cos(2.0 * theta)
    if abs(denom) < 1.0e-12:
        raise ValueError("theta is too close to a singular point of the rank-7 family")

    C = -s / SQRT2
    D = c / SQRT2
    B = c * c
    E = -s * s
    H = -s * c
    r = 1.0 / SQRT2
    b = 2.0 / denom

    U = _as_tensor(
        [
            [r, B, C, D, -C, -D, E, r],
            [0, H, D, D, -C, -C, -H, 0],
            [0, -H, -C, -C, D, D, H, 0],
            [r, E, -D, -C, D, C, B, r],
        ],
        device=device,
    )
    V, W = cyclic_factors_from_u(U)
    a = _as_tensor((split, b, b, b, b, b, b, 2.0 * SQRT2 - split), device=device)
    return U, V, W, a


def fusion_reduced(theta: float = 0.0, *, device="cpu") -> torch.Tensor:
    """Equal-split point shared by the 12-variable and rank-7 ansatzes."""
    U, _, _, a = rank7_family(theta, SQRT2, device=device)
    # At equal split this matrix has exactly the branch-A symmetry.
    A = U[0, 0]
    B = U[0, 1]
    C = U[0, 2]
    D = U[0, 3]
    E = U[0, 6]
    F = U[0, 7]
    G = U[1, 0]
    H = U[1, 1]
    I = U[1, 2]
    J = U[1, 4]
    return torch.stack((A, B, C, D, E, F, G, H, I, J, a[0], a[1]))


def rank7_theta_to_full(theta: float, split: float, *, device="cpu") -> torch.Tensor:
    U, V, W, a = rank7_family(theta, split, device=device)
    return pack(U, V, W, a)


def reduced_to_full(q: torch.Tensor) -> torch.Tensor:
    U = branch_a_u(q)
    V, W = cyclic_factors_from_u(U)
    return pack(U, V, W, branch_a_amplitudes(q))


def project_reduced(
    q0: torch.Tensor,
    *,
    tol: float = 1.0e-12,
    max_iters: int = 60,
    rcond: float = 1.0e-12,
) -> tuple[torch.Tensor, dict]:
    """Minimum-normal-displacement Newton projection onto the 3-D exact set."""
    q = q0.detach().clone()
    total_step = 0.0
    for iteration in range(max_iters + 1):
        r = branch_a_constraints(q).detach()
        rn = float(r.norm())
        if rn <= tol:
            return q, {"converged": True, "iterations": iteration, "constraint_residual": rn, "total_correction": total_step}
        if iteration == max_iters:
            break

        x = q.detach().clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(branch_a_constraints, x, vectorize=True).detach()
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        cutoff = rcond * float(S[0]) if len(S) else rcond
        keep = S > cutoff
        if not bool(keep.any()):
            break
        coeff = (U[:, keep].T @ (-r)) / S[keep]
        delta = Vh[keep, :].T @ coeff
        dn = float(delta.norm())
        if dn == 0.0:
            break

        alpha = 1.0
        accepted = False
        for _ in range(14):
            candidate = q + alpha * delta
            if float(branch_a_constraints(candidate).norm()) < rn:
                q = candidate
                total_step += alpha * dn
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    rn = float(branch_a_constraints(q).norm())
    return q, {"converged": rn <= tol, "iterations": max_iters, "constraint_residual": rn, "total_correction": total_step}


def schoolbook_to_fusion(images: int = 41, *, device="cpu") -> tuple[torch.Tensor, list[dict]]:
    """Construct a smooth exact path by projecting the straight chord.

    The chord itself is not exact.  Each point is corrected only in the normal
    directions of the 3-D algebraic solution manifold.  For this tiny problem
    the independent projections land on a smooth single branch.
    """
    start = schoolbook_reduced(device=device)
    end = fusion_reduced(0.0, device=device)
    ts = torch.linspace(0.0, 1.0, images, dtype=torch.float64, device=device)
    qs = []
    rows = []
    for i, t in enumerate(ts):
        q0 = (1.0 - t) * start + t * end
        q, info = project_reduced(q0)
        full_r = float(branch_a_full_residual(q).norm())
        qs.append(q)
        rows.append(
            {
                "image": i,
                "t": float(t),
                "constraint_residual": info["constraint_residual"],
                "full_tensor_residual": full_r,
                "correction": info["total_correction"],
                "x": float(q[10]),
                "y": float(q[11]),
            }
        )
    return torch.stack(qs), rows


def complete_homotopy(images_schoolbook: int = 41, images_transfer: int = 21, *, device="cpu") -> tuple[torch.Tensor, list[dict]]:
    """Exact schoolbook -> fusion -> Strassen+0 homotopy in full coordinates."""
    reduced, rows_a = schoolbook_to_fusion(images_schoolbook, device=device)
    full = [reduced_to_full(q) for q in reduced]
    rows = [{**r, "stage": "schoolbook_to_fusion"} for r in rows_a]

    # At theta=0 channels 0 and 7 are identical, so moving amplitude between
    # them is an exact flat direction.  Skip the first point (already present).
    for j, t in enumerate(torch.linspace(0.0, 1.0, images_transfer, dtype=torch.float64, device=device)[1:], start=1):
        split = SQRT2 + float(t) * SQRT2
        q = rank7_theta_to_full(0.0, split, device=device)
        full.append(q)
        U, V, W, a = rank7_family(0.0, split, device=device)
        target = mm_tensor(2, device)
        rr = float((torch.einsum("ir,jr,kr,r->ijk", U, V, W, a) - target).norm())
        rows.append(
            {
                "image": len(rows),
                "t": float(t),
                "constraint_residual": 0.0,
                "full_tensor_residual": rr,
                "correction": 0.0,
                "x": split,
                "y": 2.0,
                "stage": "duplicate_weight_transfer",
            }
        )
    return torch.stack(full), rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_demo(out: Path, *, device="cpu", images: int = 41) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    reduced, reduced_rows = schoolbook_to_fusion(images, device=device)
    full, full_rows = complete_homotopy(images, 21, device=device)
    torch.save(reduced.cpu(), out / "schoolbook_to_fusion_reduced.pt")
    torch.save(full.cpu(), out / "schoolbook_to_strassen_full.pt")
    write_csv(out / "schoolbook_to_fusion.csv", reduced_rows)
    write_csv(out / "schoolbook_to_strassen.csv", full_rows)

    steps = (reduced[1:] - reduced[:-1]).norm(dim=1)
    compact = torch.stack([branch_a_constraints(q).norm() for q in reduced])
    fullr = torch.stack([branch_a_full_residual(q).norm() for q in reduced])

    # Numerical Jacobian rank of the compact algebraic equations at a generic
    # interior point verifies the expected 3-D local solution manifold.
    mid = reduced[len(reduced) // 2].detach().clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(branch_a_constraints, mid, vectorize=True).detach()
    S = torch.linalg.svdvals(J)
    rank = int((S > 1.0e-10 * S[0]).sum())

    summary = {
        "reduced_variables": 12,
        "independent_polynomial_constraints": 9,
        "generic_jacobian_rank": rank,
        "generic_manifold_dimension": 12 - rank,
        "schoolbook_to_fusion_images": images,
        "max_compact_constraint_residual": float(compact.max()),
        "max_full_tensor_residual": float(fullr.max()),
        "max_reduced_step": float(steps.max()),
        "max_projection_correction": max(r["correction"] for r in reduced_rows),
        "full_homotopy_max_tensor_residual": max(r["full_tensor_residual"] for r in full_rows),
        "final_zero_channel_amplitude": 0.0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Low-dimensional exact ansatz for the 2x2 schoolbook-to-Strassen path")
    p.add_argument("--mode", choices=("demo", "rank7"), default="demo")
    p.add_argument("--out", type=Path, default=Path("runs/ansatz"))
    p.add_argument("--images", type=int, default=41)
    p.add_argument("--theta", type=float, default=0.0)
    p.add_argument("--split", type=float, default=2.0 * SQRT2)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.mode == "demo":
        run_demo(args.out, device=args.device, images=args.images)
    else:
        U, V, W, a = rank7_family(args.theta, args.split, device=args.device)
        target = mm_tensor(2, args.device)
        residual = float((torch.einsum("ir,jr,kr,r->ijk", U, V, W, a) - target).norm())
        print("residual:", residual)
        print("amplitudes:", [float(x) for x in a])


if __name__ == "__main__":
    main()
