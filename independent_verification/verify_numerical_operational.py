#!/usr/bin/env python3
"""Independent numerical operational verifier for a rank-R checkpoint.

This intentionally does NOT import any project code or construct the target tensor.
It deserializes U,V,W,a and literally executes the bilinear algorithm on matrices,
then compares with ordinary matrix multiplication.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch


def load_checkpoint(path: Path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("U", "V", "W"):
        if k not in d:
            raise KeyError(f"checkpoint missing {k}")
    U = np.asarray(d["U"].detach().cpu(), dtype=np.float64)
    V = np.asarray(d["V"].detach().cpu(), dtype=np.float64)
    W = np.asarray(d["W"].detach().cpu(), dtype=np.float64)
    if "a" in d:
        c = np.asarray(d["a"].detach().cpu(), dtype=np.float64)
    elif "c" in d:
        c = np.asarray(d["c"].detach().cpu(), dtype=np.float64)
    else:
        c = np.ones(U.shape[1], dtype=np.float64)
    if U.shape[0] != 9 or V.shape[0] != 9 or W.shape[0] != 9:
        raise ValueError(f"expected 9xR factors, got {U.shape}, {V.shape}, {W.shape}")
    if not (U.shape[1] == V.shape[1] == W.shape[1] == len(c)):
        raise ValueError("rank dimensions disagree")
    return U, V, W, c


def fast_mm(A, B, U, V, W, c):
    af = np.asarray(A, dtype=np.float64).reshape(9)
    bf = np.asarray(B, dtype=np.float64).reshape(9)
    left = U.T @ af
    right = V.T @ bf
    products = c * left * right
    return (W @ products).reshape(3, 3)


def fast_mm_blocks(A, B, U, V, W, c):
    # A,B shape (3,3,2,2); entries live in the noncommutative ring M_2(R).
    R = U.shape[1]
    products = []
    for r in range(R):
        L = np.zeros((2,2), dtype=np.float64)
        M = np.zeros((2,2), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                q = 3*i+j
                L += U[q,r] * A[i,j]
                M += V[q,r] * B[i,j]
        products.append(c[r] * (L @ M))
    C = np.zeros((3,3,2,2), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            q = 3*i+j
            for r in range(R):
                C[i,j] += W[q,r] * products[r]
    return C


def naive_blocks(A, B):
    C = np.zeros((3,3,2,2), dtype=np.float64)
    for i in range(3):
        for k in range(3):
            for j in range(3):
                C[i,k] += A[i,j] @ B[j,k]
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--nc-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args()
    U,V,W,c = load_checkpoint(args.checkpoint)
    rng = np.random.default_rng(args.seed)

    max_basis = 0.0
    for qa in range(9):
        for qb in range(9):
            A = np.zeros((3,3)); B = np.zeros((3,3))
            A.flat[qa] = 1.0; B.flat[qb] = 1.0
            err = np.max(np.abs(fast_mm(A,B,U,V,W,c) - A@B))
            max_basis = max(max_basis, float(err))

    max_abs = 0.0; max_rel = 0.0
    for _ in range(args.trials):
        A = rng.normal(size=(3,3)); B = rng.normal(size=(3,3))
        got = fast_mm(A,B,U,V,W,c); ref = A@B
        e = np.max(np.abs(got-ref))
        rel = e / max(1.0, np.max(np.abs(ref)))
        max_abs = max(max_abs, float(e)); max_rel = max(max_rel, float(rel))

    max_nc = 0.0
    for _ in range(args.nc_trials):
        A = rng.normal(size=(3,3,2,2)); B = rng.normal(size=(3,3,2,2))
        e = np.max(np.abs(fast_mm_blocks(A,B,U,V,W,c) - naive_blocks(A,B)))
        max_nc = max(max_nc, float(e))

    ok = max(max_basis, max_rel, max_nc) <= args.tol
    print("# independent numerical operational verification")
    print(f"rank: {len(c)}")
    print(f"basis-pair max abs error: {max_basis:.3e}")
    print(f"random scalar max abs error: {max_abs:.3e}")
    print(f"random scalar max relative error: {max_rel:.3e}")
    print(f"noncommutative M2(R) max abs error: {max_nc:.3e}")
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

if __name__ == "__main__":
    main()
