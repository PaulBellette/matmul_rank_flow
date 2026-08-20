"""Exactification and inspection tools for a numerically discovered rank-23 scheme.

This is intentionally independent of the search controller.  Given a saved
``final.pt`` checkpoint it:

1. verifies and tensor-polishes the numerical CP decomposition;
2. fixes the per-channel CP scaling gauge using deterministic pivots;
3. tries simple rational/algebraic recognition of every coefficient;
4. verifies any recognised candidate against the 729 Brent identities *exactly*;
5. optionally compares directly with the published ternary rank-23 reference
   up to channel permutation + per-channel scalar gauge.

The direct reference comparison is deliberately conservative: it does not try
the full isotropy group of matrix multiplication (global GL(3) basis changes).
A failed direct match therefore does *not* imply a genuinely inequivalent
rank-23 family.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
import torch
from scipy.optimize import linear_sum_assignment

from geometry_flow import (
    gauss_newton_correct,
    pack,
    residual_vector,
    unit_columns,
    unpack,
)
from rankflow import mm_tensor


torch.set_default_dtype(torch.float64)


@dataclass
class RecognitionStats:
    mode: str
    total_scalars: int
    recognised_scalars: int
    max_abs_error: float
    numeric_tensor_residual: float
    exact_identity: bool


@dataclass
class DirectMatchStats:
    mean_cost: float
    max_cost: float
    matched_channels: list[tuple[int, int, float]]


def load_checkpoint(path: Path) -> tuple[torch.Tensor, int]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    rank = int(obj.get("rank", 0)) if isinstance(obj, dict) else 0
    if isinstance(obj, dict) and "theta" in obj:
        q = obj["theta"].detach().clone().to(torch.float64)
        if not rank:
            # theta has (3*n^2 + 1)R entries, here n=3 -> 28R.
            if q.numel() % 28:
                raise ValueError("cannot infer rank from theta length")
            rank = q.numel() // 28
        return q, rank
    if isinstance(obj, dict) and all(k in obj for k in ("U", "V", "W", "a")):
        U = obj["U"].to(torch.float64)
        V = obj["V"].to(torch.float64)
        W = obj["W"].to(torch.float64)
        a = obj["a"].to(torch.float64)
        return pack(U, V, W, a), int(a.numel())
    if torch.is_tensor(obj):
        q = obj.to(torch.float64)
        if q.numel() % 28:
            raise ValueError("cannot infer rank from tensor length")
        return q, q.numel() // 28
    raise TypeError(f"unsupported checkpoint format in {path}")


def canonical_channel_gauge(theta: torch.Tensor, n: int, rank: int):
    """Return factor columns with deterministic pivot=+1 and scalar weights.

    Reconstruction normalises U/V/W internally.  First make that gauge explicit,
    then divide each factor by its largest-magnitude signed pivot.  The product
    of those three pivot values is absorbed into the channel coefficient.
    """
    U, V, W, a = unpack(theta, n, rank)
    U, V, W = unit_columns(U), unit_columns(V), unit_columns(W)
    Uc, Vc, Wc = U.clone(), V.clone(), W.clone()
    c = a.clone()
    pivots: list[tuple[int, int, int]] = []
    for r in range(rank):
        cols = []
        idxs = []
        for X in (Uc, Vc, Wc):
            idx = int(torch.argmax(X[:, r].abs()))
            p = float(X[idx, r])
            if abs(p) < 1e-14:
                raise ValueError(f"zero pivot in channel {r}")
            X[:, r] /= p
            c[r] *= p
            idxs.append(idx)
        pivots.append(tuple(idxs))
    return Uc, Vc, Wc, c, pivots


def tensor_from_canonical(U: torch.Tensor, V: torch.Tensor, W: torch.Tensor, c: torch.Tensor):
    return torch.einsum("ir,jr,kr,r->ijk", U, V, W, c)


def rational_recognise(x: float, max_denominator: int, tol: float) -> tuple[sp.Expr, float, bool]:
    f = Fraction(float(x)).limit_denominator(max_denominator)
    err = abs(float(f) - float(x))
    return sp.Rational(f.numerator, f.denominator), err, err <= tol


def algebraic_recognise(x: float, tol: float) -> tuple[sp.Expr, float, bool]:
    constants = [sp.sqrt(2), sp.sqrt(3), sp.sqrt(5), sp.sqrt(6), sp.sqrt(7), sp.sqrt(10)]
    expr = sp.nsimplify(float(x), constants, tolerance=tol, full=False)
    err = abs(float(sp.N(expr, 30)) - float(x))
    # Avoid accepting a giant rational that merely encodes the binary float.
    complexity = len(str(expr))
    ok = err <= tol and complexity <= 48
    return expr, err, ok


def recognise_arrays(
    arrays: list[torch.Tensor],
    *,
    mode: str,
    max_denominator: int,
    tol: float,
):
    recognised: list[list[list[sp.Expr]] | list[sp.Expr]] = []
    n_ok = 0
    total = 0
    max_err = 0.0
    for arr in arrays:
        a = arr.detach().cpu().numpy()
        out = np.empty(a.shape, dtype=object)
        for idx in np.ndindex(a.shape):
            x = float(a[idx])
            if mode == "rational":
                expr, err, ok = rational_recognise(x, max_denominator, tol)
            elif mode == "algebraic":
                expr, err, ok = algebraic_recognise(x, tol)
            else:
                raise ValueError(mode)
            out[idx] = expr
            total += 1
            n_ok += int(ok)
            max_err = max(max_err, err)
        recognised.append(out.tolist())
    return recognised, n_ok, total, max_err


def exact_mm_target(n: int = 3):
    d = n * n
    out = [[[sp.Integer(0) for _ in range(d)] for _ in range(d)] for _ in range(d)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                out[i*n+j][j*n+k][i*n+k] = sp.Integer(1)
    return out


def exact_verify(U, V, W, c, n: int = 3) -> tuple[bool, int, list[str]]:
    d = n * n
    target = exact_mm_target(n)
    failures: list[str] = []
    nonzero = 0
    rank = len(c)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                val = sp.Integer(0)
                for r in range(rank):
                    val += c[r] * U[i][r] * V[j][r] * W[k][r]
                diff = sp.factor(val - target[i][j][k])
                if diff != 0:
                    nonzero += 1
                    if len(failures) < 20:
                        failures.append(f"({i},{j},{k}): {diff}")
    return nonzero == 0, nonzero, failures


def numeric_from_exprs(U, V, W, c) -> float:
    Un = torch.tensor([[float(sp.N(x, 30)) for x in row] for row in U], dtype=torch.float64)
    Vn = torch.tensor([[float(sp.N(x, 30)) for x in row] for row in V], dtype=torch.float64)
    Wn = torch.tensor([[float(sp.N(x, 30)) for x in row] for row in W], dtype=torch.float64)
    cn = torch.tensor([float(sp.N(x, 30)) for x in c], dtype=torch.float64)
    return float((tensor_from_canonical(Un, Vn, Wn, cn) - mm_tensor(3)).norm())


def expr_to_strings(obj: Any):
    if isinstance(obj, list):
        return [expr_to_strings(x) for x in obj]
    return str(obj)


def canonical_reference():
    from rank23_reference import reference_theta
    return canonical_channel_gauge(reference_theta(), 3, 23)[:4]


def direct_reference_match(U, V, W, c) -> DirectMatchStats:
    Ur, Vr, Wr, cr = canonical_reference()
    R = c.numel()
    if R != 23:
        raise ValueError("direct reference comparison only makes sense at rank 23")
    cost = np.zeros((R, R), dtype=float)
    for i in range(R):
        for j in range(R):
            diffs = [
                float((U[:, i] - Ur[:, j]).abs().max()),
                float((V[:, i] - Vr[:, j]).abs().max()),
                float((W[:, i] - Wr[:, j]).abs().max()),
                abs(float(c[i] - cr[j])),
            ]
            cost[i, j] = max(diffs)
    rr, cc = linear_sum_assignment(cost)
    pairs = [(int(i), int(j), float(cost[i, j])) for i, j in zip(rr, cc)]
    vals = [p[2] for p in pairs]
    return DirectMatchStats(float(np.mean(vals)), float(np.max(vals)), pairs)


def write_markdown(path: Path, report: dict):
    rec = report.get("recognition", {})
    direct = report.get("direct_reference_match")
    lines = [
        "# Rank-23 exactification report",
        "",
        f"- input: `{report['input']}`",
        f"- rank: {report['rank']}",
        f"- input tensor residual: `{report['input_residual']:.6e}`",
        f"- refined tensor residual: `{report['refined_residual']:.6e}`",
        f"- max |amplitude| after refinement: `{report['max_abs_amplitude']:.6f}`",
        "",
    ]
    if rec:
        lines += ["## Coefficient recognition", ""]
        for mode, stats in rec.items():
            lines += [
                f"### {mode}",
                "",
                f"- recognised scalars: {stats['recognised_scalars']} / {stats['total_scalars']}",
                f"- max scalar approximation error: `{stats['max_abs_error']:.3e}`",
                f"- candidate tensor residual: `{stats['numeric_tensor_residual']:.3e}`",
                f"- exact 729-identity verification: **{stats['exact_identity']}**",
                "",
            ]
    if direct is not None:
        lines += [
            "## Direct published-reference comparison",
            "",
            "This only quotients channel permutation and per-channel CP scaling gauge; it does not search the full matrix-multiplication isotropy group.",
            "",
            f"- mean matched-channel cost: `{direct['mean_cost']:.3e}`",
            f"- max matched-channel cost: `{direct['max_cost']:.3e}`",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/exactify_rank23"))
    ap.add_argument("--refine-iters", type=int, default=20)
    ap.add_argument("--refine-tol", type=float, default=2e-14)
    ap.add_argument("--rcond", type=float, default=1e-10)
    ap.add_argument("--recognition-tol", type=float, default=2e-7)
    ap.add_argument("--max-denominator", type=int, default=64)
    ap.add_argument("--compare-reference", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    q0, rank = load_checkpoint(args.checkpoint)
    if rank != 23:
        raise SystemExit(f"expected rank 23 checkpoint, got rank {rank}")
    T = mm_tensor(3)
    initial_res = float(residual_vector(q0, T, 3, rank).norm())

    q, info = gauss_newton_correct(
        q0, T, 3, rank,
        tol=args.refine_tol,
        max_iters=args.refine_iters,
        rcond=args.rcond,
    )
    refined_res = float(residual_vector(q, T, 3, rank).norm())
    U, V, W, c, pivots = canonical_channel_gauge(q, 3, rank)

    # Save the independently polished numerical checkpoint immediately.
    torch.save(
        {"theta": q, "rank": rank, "U": unpack(q, 3, rank)[0], "V": unpack(q, 3, rank)[1],
         "W": unpack(q, 3, rank)[2], "a": unpack(q, 3, rank)[3]},
        args.out / "rank23_refined.pt",
    )

    report: dict[str, Any] = {
        "input": str(args.checkpoint),
        "rank": rank,
        "input_residual": initial_res,
        "refined_residual": refined_res,
        "refine_converged": bool(info.converged),
        "refine_iterations": int(info.iterations),
        "max_abs_amplitude": float(unpack(q, 3, rank)[3].abs().max()),
        "pivots": pivots,
        "recognition": {},
    }

    for mode in ("rational", "algebraic"):
        exprs, n_ok, total, max_err = recognise_arrays(
            [U, V, W, c], mode=mode, max_denominator=args.max_denominator, tol=args.recognition_tol
        )
        Ue, Ve, We, ce = exprs
        numeric_res = numeric_from_exprs(Ue, Ve, We, ce)
        exact = False
        failures: list[str] = []
        if n_ok == total:
            exact, nonzero, failures = exact_verify(Ue, Ve, We, ce, 3)
        else:
            nonzero = -1
        stats = RecognitionStats(mode, total, n_ok, max_err, numeric_res, exact)
        report["recognition"][mode] = asdict(stats) | {"nonzero_exact_identities": nonzero, "first_failures": failures}
        cert = {
            "mode": mode,
            "rank": rank,
            "U": expr_to_strings(Ue),
            "V": expr_to_strings(Ve),
            "W": expr_to_strings(We),
            "c": expr_to_strings(ce),
            "exact_identity": exact,
            "recognised_scalars": n_ok,
            "total_scalars": total,
        }
        (args.out / f"candidate_{mode}.json").write_text(json.dumps(cert, indent=2) + "\n")

    if args.compare_reference:
        match = direct_reference_match(U, V, W, c)
        report["direct_reference_match"] = asdict(match)

    (args.out / "exactify_report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(args.out / "EXACTIFY_REPORT.md", report)

    print(f"input residual:   {initial_res:.3e}")
    print(f"refined residual: {refined_res:.3e}  converged={info.converged}")
    for mode, stats in report["recognition"].items():
        print(
            f"{mode:9s}: recognised {stats['recognised_scalars']}/{stats['total_scalars']} "
            f"candidate_res={stats['numeric_tensor_residual']:.3e} exact={stats['exact_identity']}"
        )
    if "direct_reference_match" in report:
        m = report["direct_reference_match"]
        print(f"direct reference match: mean={m['mean_cost']:.3e} max={m['max_cost']:.3e}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
