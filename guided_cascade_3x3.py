"""Diagnostics for the endpoint-guided 27 -> 23 cascade.

This does *not* claim an autonomous 3x3 discovery.  The first 27 -> 26 step is
local collision geometry.  The later branch was located using an independent
exact rank-23 endpoint as a global guide, then converted into finite-coefficient
rank-25/rank-24/rank-23 checkpoints.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from geometry_flow import pack, unpack, residual_vector, pinv_solve
from curvature_flow import physical_constraints
from rankflow import mm_tensor

ROOT = Path(__file__).resolve().parent
REF = ROOT / "reference_guided_cascade"

torch.set_default_dtype(torch.float64)


def load_theta(path: Path, rank: int) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if "theta" in obj:
        return obj["theta"]
    return pack(obj["U"], obj["V"], obj["W"], obj["a"])


def checkpoint_report() -> list[tuple[int, float, float, float]]:
    T = mm_tensor(3)
    rows = []
    files = {26: "rank26.pt", 25: "rank25.pt", 24: "rank24.pt", 23: "rank23_flow.pt"}
    for rank, name in files.items():
        q = load_theta(REF / name, rank)
        a = unpack(q, 3, rank)[3]
        rows.append((rank, float(residual_vector(q, T, 3, rank).norm()), float(a.abs().min()), float(a.abs().max())))
    return rows


def continue_last_drop(steps: int = 9) -> torch.Tensor:
    """Reproduce the regular rank-24 -> rank-23 amplitude continuation."""
    T = mm_tensor(3)
    rank = 24
    channel = 9
    q = load_theta(REF / "rank24.pt", rank).clone()
    amp_offset = 3 * 3 * 3 * rank + channel
    start = float(q[amp_offset])

    def constraints(x: torch.Tensor, target_amp: float) -> torch.Tensor:
        return torch.cat([physical_constraints(x, T, 3, rank), (x[amp_offset] - target_amp).reshape(1)])

    def jacobian(x: torch.Tensor, target_amp: float) -> torch.Tensor:
        z = x.detach().clone().requires_grad_(True)
        return torch.autograd.functional.jacobian(lambda y: constraints(y, target_amp), z, vectorize=True).detach()

    for target_amp in torch.linspace(start, 0.0, steps + 1)[1:]:
        target_amp = float(target_amp)
        for _ in range(12):
            r = constraints(q, target_amp).detach()
            if float(r.norm()) < 1e-10:
                break
            J = jacobian(q, target_amp)
            delta = pinv_solve(J, -r, rcond=1e-8)
            alpha = 1.0
            base = float(r.norm())
            for _ in range(15):
                candidate = q + alpha * delta
                if float(constraints(candidate, target_amp).norm()) < base:
                    q = candidate
                    break
                alpha *= 0.5
        print(f"a[{channel}]={float(q[amp_offset]): .6e}  tensor residual={float(residual_vector(q,T,3,rank).norm()):.3e}")

    U, V, W, a = unpack(q, 3, rank)
    keep = [i for i in range(rank) if i != channel]
    return pack(U[:, keep], V[:, keep], W[:, keep], a[keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["verify", "last-drop"], default="verify")
    args = ap.parse_args()
    if args.mode == "verify":
        print("rank  residual         min|a|          max|a|")
        for rank, res, amin, amax in checkpoint_report():
            print(f"{rank:4d}  {res: .3e}   {amin: .6f}   {amax: .6f}")
    else:
        q23 = continue_last_drop()
        print(f"rank-23 residual after drop: {float(residual_vector(q23, mm_tensor(3), 3, 23).norm()):.3e}")


if __name__ == "__main__":
    main()
