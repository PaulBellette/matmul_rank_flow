from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from analytic_ansatz import (
    SQRT2,
    branch_a_amplitudes,
    branch_a_constraints,
    branch_a_full_residual,
    branch_a_u,
    cyclic_factors_from_u,
    rank7_family,
    reduced_to_full,
)
from geometry_flow import pack
from rankflow import mm_tensor


torch.set_default_dtype(torch.float64)

S_FUSION = 1.0 / SQRT2


def reduced_from_collision_p(p: float, *, device: str = "cpu") -> torch.Tensor:
    """Closed-form schoolbook -> duplicate-channel fusion branch.

    p = A*F = D^2 lies in [0, 1/2].  In the discovered symmetry sector we set

        B = 1,
        C = E = G = H = J = 0,

    and solve the exact tensor equations analytically.  The positive branch is

        A+F = sqrt(1+2p),
        A-F = sqrt(1-2p),
        D^2 = p,
        I^2 = 1-p,
        y = 1/(1-p),
        x = 1/((1-p)*sqrt(1+2p)).

    p=0 is schoolbook.  p=1/2 is the equal-split fusion point where channels
    0 and 7 are identical rank-one tensors.
    """
    if p < -1.0e-14 or p > 0.5 + 1.0e-14:
        raise ValueError("p must lie in [0, 1/2]")
    p = min(0.5, max(0.0, float(p)))

    plus = math.sqrt(1.0 + 2.0 * p)
    minus = math.sqrt(max(0.0, 1.0 - 2.0 * p))
    A = 0.5 * (plus + minus)
    F = 0.5 * (plus - minus)
    D = math.sqrt(p)
    I = math.sqrt(1.0 - p)
    x = 1.0 / ((1.0 - p) * plus)
    y = 1.0 / (1.0 - p)

    return torch.tensor(
        [A, 1.0, 0.0, D, 0.0, F, 0.0, 0.0, I, 0.0, x, y],
        dtype=torch.float64,
        device=device,
    )


def reduced_from_s(s: float, *, device: str = "cpu") -> torch.Tensor:
    """Smooth schoolbook-end parameterization with p=s^2.

    s in [0, 1/sqrt(2)].  Near s=0 the path has the expansions

        D = s,
        F = s^2 + O(s^6),
        I = 1 - s^2/2 + O(s^4),
        y = 1 + s^2 + O(s^4),
        x = 1 + 3 s^4/2 + O(s^6).

    Thus the exact path initially moves sideways in factor space while every
    channel amplitude remains stationary to first order.  This is the local
    rigidity seen in the Jacobian calculations.
    """
    if s < -1.0e-14 or s > S_FUSION + 1.0e-14:
        raise ValueError("s must lie in [0, 1/sqrt(2)]")
    s = min(S_FUSION, max(0.0, float(s)))
    p = 0.5 if abs(s - S_FUSION) < 1.0e-12 else s * s
    return reduced_from_collision_p(p, device=device)


def factors_from_s(s: float, *, device: str = "cpu"):
    q = reduced_from_s(s, device=device)
    U = branch_a_u(q)
    V, W = cyclic_factors_from_u(U)
    a = branch_a_amplitudes(q)
    return U, V, W, a


def channel_linear_forms(s: float):
    """Return the particularly simple factor columns on the closed branch.

    Flattening convention is [11, 12, 21, 22].  The return value is a list of
    dictionaries containing u, v, w and the scalar amplitude for each of the
    eight rank-one terms.
    """
    U, V, W, a = factors_from_s(s)
    rows = []
    for r in range(8):
        rows.append(
            {
                "channel": r,
                "amplitude": float(a[r]),
                "u": [float(v) for v in U[:, r]],
                "v": [float(v) for v in V[:, r]],
                "w": [float(v) for v in W[:, r]],
            }
        )
    return rows


def collision_metrics(s: float) -> dict:
    """Geometry of the two channels that ultimately fuse.

    The columns are unit vectors.  Along this branch

        <u0,u7> = <v0,v7> = <w0,w7> = 2 s^2.

    Therefore their factor cosine similarity grows monotonically 0 -> 1.
    The cosine similarity of the full normalized rank-one tensors is its cube.
    """
    U, V, W, a = factors_from_s(s)
    cu = float(torch.dot(U[:, 0], U[:, 7]))
    cv = float(torch.dot(V[:, 0], V[:, 7]))
    cw = float(torch.dot(W[:, 0], W[:, 7]))
    return {
        "s": float(s),
        "p": float(s * s),
        "u_cosine": cu,
        "v_cosine": cv,
        "w_cosine": cw,
        "rank1_cosine": cu * cv * cw,
        "factor_distance": float((U[:, 0] - U[:, 7]).norm()),
        "a0": float(a[0]),
        "a7": float(a[7]),
        "common_six_amplitude": float(a[1]),
    }


def closed_form_tensor_residual(s: float, *, device: str = "cpu") -> float:
    U, V, W, a = factors_from_s(s, device=device)
    T = torch.einsum("ir,jr,kr,r->ijk", U, V, W, a)
    return float((T - mm_tensor(2, device)).norm())


def transfer_from_fusion(t: float, *, device: str = "cpu"):
    """Exact duplicate-weight transfer, t in [0,1].

    At the fusion point channels 0 and 7 are identical.  Their equal starting
    weights are sqrt(2), sqrt(2).  Transfer weight until they are
    2*sqrt(2), 0, yielding a seven-product Strassen decomposition.
    """
    if t < -1.0e-14 or t > 1.0 + 1.0e-14:
        raise ValueError("t must lie in [0,1]")
    t = min(1.0, max(0.0, float(t)))
    split = SQRT2 * (1.0 + t)
    return rank7_family(0.0, split, device=device)


def full_homotopy(t: float, *, device: str = "cpu") -> torch.Tensor:
    """Piecewise exact schoolbook -> fusion -> Strassen homotopy, t in [0,1].

    First half follows the closed-form rank-8 collision branch, using s as a
    smooth coordinate.  Second half transfers weight between the duplicated
    channels after they have fused.
    """
    if t < -1.0e-14 or t > 1.0 + 1.0e-14:
        raise ValueError("t must lie in [0,1]")
    t = min(1.0, max(0.0, float(t)))
    if t <= 0.5:
        s = 2.0 * t * S_FUSION
        return reduced_to_full(reduced_from_s(s, device=device))
    U, V, W, a = transfer_from_fusion(2.0 * t - 1.0, device=device)
    return pack(U, V, W, a)


def sample_closed_form(images: int = 81, *, device: str = "cpu"):
    rows = []
    points = []
    for i, t in enumerate(torch.linspace(0.0, 1.0, images, dtype=torch.float64)):
        tf = float(t)
        point = full_homotopy(tf, device=device)
        points.append(point)
        if tf <= 0.5:
            s = 2.0 * tf * S_FUSION
            q = reduced_from_s(s, device=device)
            m = collision_metrics(s)
            residual = float(branch_a_full_residual(q).norm())
            rows.append(
                {
                    "image": i,
                    "t": tf,
                    "stage": "collision",
                    "s": s,
                    "p": s * s,
                    "residual": residual,
                    "a0": float(q[10]),
                    "a7": float(q[10]),
                    "a_six": float(q[11]),
                    "factor_cosine_0_7": m["u_cosine"],
                    "rank1_cosine_0_7": m["rank1_cosine"],
                }
            )
        else:
            transfer_t = 2.0 * tf - 1.0
            U, V, W, a = transfer_from_fusion(transfer_t, device=device)
            residual = float(
                (torch.einsum("ir,jr,kr,r->ijk", U, V, W, a) - mm_tensor(2, device)).norm()
            )
            rows.append(
                {
                    "image": i,
                    "t": tf,
                    "stage": "weight_transfer",
                    "s": S_FUSION,
                    "p": 0.5,
                    "residual": residual,
                    "a0": float(a[0]),
                    "a7": float(a[7]),
                    "a_six": float(a[1]),
                    "factor_cosine_0_7": 1.0,
                    "rank1_cosine_0_7": 1.0,
                }
            )
    return torch.stack(points), rows


def write_algorithm_text(path: Path, s: float) -> None:
    q = reduced_from_s(s)
    A, _, _, D, _, F, _, _, I, _, x, y = [float(v) for v in q]
    text = "# Eight-product algorithm on the closed homotopy\n\n"
    text += f"s = {s:.16g}\n\n"
    text += f"A = {A:.16g}\nF = {F:.16g}\nD = {D:.16g}\nI = {I:.16g}\nx = {x:.16g}\ny = {y:.16g}\n\n"
    text += "Flattening order is [11,12,21,22].  Each channel computes\n"
    text += "m_r = amplitude * <u_r,A> * <v_r,B>, then adds m_r*w_r to C.\n\n"
    for row in channel_linear_forms(s):
        text += f"channel {row['channel']}: a={row['amplitude']:.16g}\n"
        text += f"  u={row['u']}\n  v={row['v']}\n  w={row['w']}\n"
    path.write_text(text)


def run_demo(out: Path, images: int = 81) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    points, rows = sample_closed_form(images)
    torch.save(points.cpu(), out / "closed_form_homotopy.pt")
    with (out / "closed_form_homotopy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_algorithm_text(out / "schoolbook.txt", 0.0)
    write_algorithm_text(out / "fusion.txt", S_FUSION)

    # Verify compact polynomial equations over the entire collision stage.
    ss = torch.linspace(0.0, S_FUSION, max(3, images // 2 + 1))
    compact = []
    full = []
    for s in ss:
        q = reduced_from_s(float(s))
        compact.append(float(branch_a_constraints(q).norm()))
        full.append(float(branch_a_full_residual(q).norm()))

    start_collision = collision_metrics(0.0)
    end_collision = collision_metrics(S_FUSION)
    summary = {
        "closed_form_collision_parameter": "p=s^2 in [0,1/2]",
        "images": images,
        "max_compact_constraint_residual": max(compact),
        "max_full_tensor_residual_collision_stage": max(full),
        "max_full_tensor_residual_complete_homotopy": max(r["residual"] for r in rows),
        "schoolbook_factor_cosine_0_7": start_collision["u_cosine"],
        "fusion_factor_cosine_0_7": end_collision["u_cosine"],
        "schoolbook_amplitudes": [1.0] * 8,
        "fusion_equal_split_amplitudes": [SQRT2] + [2.0] * 6 + [SQRT2],
        "strassen_final_amplitudes": [2.0 * SQRT2] + [2.0] * 6 + [0.0],
    }
    (out / "summary_closed_form.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-form schoolbook-to-Strassen homotopy")
    p.add_argument("--mode", choices=("demo", "point", "algorithm"), default="demo")
    p.add_argument("--out", type=Path, default=Path("runs/closed_form"))
    p.add_argument("--images", type=int, default=81)
    p.add_argument("--s", type=float, default=0.0)
    args = p.parse_args()

    if args.mode == "demo":
        run_demo(args.out, args.images)
    elif args.mode == "point":
        q = reduced_from_s(args.s)
        print("q:", [float(v) for v in q])
        print("compact residual:", float(branch_a_constraints(q).norm()))
        print("full tensor residual:", float(branch_a_full_residual(q).norm()))
        print("collision:", json.dumps(collision_metrics(args.s), indent=2))
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        target = args.out / "algorithm.txt"
        write_algorithm_text(target, args.s)
        print(target)


if __name__ == "__main__":
    main()
