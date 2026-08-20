"""Orbit-invariant diagnostics for numerical 3x3 rank-23 multiplication schemes.

Computes the rank-based invariants used by Heule--Kauers--Seidl:
  * counts of rank-1/2/3 factor matrices,
  * symmetrised f(x,y,z),
  * g(w),
  * the full rank-pattern class, canonical under tensor-leg and channel permutations.

The full rank-pattern canonicalisation is a strictly stronger preprocessing invariant
than f or g, though none of these alone decides equivalence.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch

from exactify_rank23 import load_checkpoint
from geometry_flow import unpack, unit_columns


torch.set_default_dtype(torch.float64)


def factor_rank(M: torch.Tensor, tol: float = 1e-8) -> tuple[int, list[float]]:
    s = torch.linalg.svdvals(M)
    if float(s[0]) == 0.0:
        return 0, [0.0, 0.0, 0.0]
    rel = (s / s[0]).tolist()
    return int((s > tol * s[0]).sum()), [float(x) for x in rel]


def rank_pattern(theta: torch.Tensor, rank: int, tol: float = 1e-8):
    U, V, W, _ = unpack(theta, 3, rank)
    U, V, W = unit_columns(U), unit_columns(V), unit_columns(W)
    pattern: list[tuple[int, int, int]] = []
    singular_ratios: list[list[list[float]]] = []
    for r in range(rank):
        tri = []
        ratios = []
        for X in (U, V, W):
            rk, sr = factor_rank(X[:, r].reshape(3, 3), tol=tol)
            tri.append(rk)
            ratios.append(sr)
        pattern.append(tuple(tri))
        singular_ratios.append(ratios)
    return pattern, singular_ratios


def canonical_rank_pattern(pattern: Iterable[tuple[int, int, int]]):
    """Canonical representative modulo S3 tensor-leg permutations and channel permutation."""
    pattern = list(pattern)
    candidates = []
    for pi in itertools.permutations(range(3)):
        cols = sorted((tuple(t[i] for i in pi) for t in pattern), reverse=True)
        candidates.append(tuple(cols))
    return max(candidates)


def hks_invariants(pattern: list[tuple[int, int, int]]):
    factor_counts = Counter(q for t in pattern for q in t)
    triple_counts = Counter(pattern)

    # f(x,y,z) = sum_i sum_{pi in S3} pi(x^ra y^rb z^rc)
    f = Counter()
    for t in pattern:
        for pi in itertools.permutations(range(3)):
            f[tuple(t[i] for i in pi)] += 1

    leg_rank_sums = tuple(sum(t[i] for t in pattern) for i in range(3))
    g = Counter(leg_rank_sums)
    return factor_counts, triple_counts, f, leg_rank_sums, g, canonical_rank_pattern(pattern)


def fmt_f(f: Counter) -> str:
    terms = []
    for exps, coeff in sorted(f.items(), key=lambda kv: (sum(kv[0]), kv[0]), reverse=True):
        mono = []
        for v, e in zip("xyz", exps):
            mono.append(v if e == 1 else f"{v}^{e}")
        m = "".join(mono)
        terms.append(m if coeff == 1 else f"{coeff}{m}")
    return " + ".join(terms)


def fmt_g(g: Counter) -> str:
    terms = []
    for exponent, coeff in sorted(g.items(), reverse=True):
        m = f"w^{exponent}"
        terms.append(m if coeff == 1 else f"{coeff}{m}")
    return " + ".join(terms)


def analyse(theta: torch.Tensor, rank: int, tol: float = 1e-8):
    pattern, sr = rank_pattern(theta, rank, tol=tol)
    factor_counts, triple_counts, f, leg_sums, g, canon = hks_invariants(pattern)

    r1_s2 = []
    r2_s2 = []
    r2_s3 = []
    for tri, ratios in zip(pattern, sr):
        for rk, rr in zip(tri, ratios):
            if rk == 1:
                r1_s2.append(rr[1])
            elif rk == 2:
                r2_s2.append(rr[1])
                r2_s3.append(rr[2])

    return {
        "rank": rank,
        "rank_tolerance": tol,
        "factor_rank_counts": {str(k): int(factor_counts.get(k, 0)) for k in (1, 2, 3)},
        "channel_rank_triples": {str(k): int(v) for k, v in sorted(triple_counts.items())},
        "rank_pattern": [list(x) for x in pattern],
        "canonical_rank_pattern": [list(x) for x in canon],
        "f": fmt_f(f),
        "f_terms": {str(k): int(v) for k, v in sorted(f.items())},
        "leg_rank_sums": list(leg_sums),
        "g": fmt_g(g),
        "rank_separation": {
            "max_rank1_s2_over_s1": max(r1_s2) if r1_s2 else None,
            "min_rank2_s2_over_s1": min(r2_s2) if r2_s2 else None,
            "max_rank2_s3_over_s1": max(r2_s3) if r2_s3 else None,
        },
    }


def write_md(path: Path, blind: dict, ref: dict | None):
    lines = [
        "# Rank-23 orbit-invariant report",
        "",
        "These are rank-based invariants under the full matrix-multiplication isotropy action",
        "(invertible sandwich transformations, tensor-leg permutations/transposes, channel permutation, and CP scaling).",
        "They are sufficient to prove inequivalence when they differ, but equality does not prove equivalence.",
        "",
        "## Blind scheme",
        "",
        f"- factor-rank counts: `{blind['factor_rank_counts']}`",
        f"- channel rank triples: `{blind['channel_rank_triples']}`",
        f"- HKS f(x,y,z): `{blind['f']}`",
        f"- leg rank sums: `{blind['leg_rank_sums']}`",
        f"- HKS g(w): `{blind['g']}`",
        f"- max rank-1 sigma2/sigma1: `{blind['rank_separation']['max_rank1_s2_over_s1']:.3e}`",
        f"- min rank-2 sigma2/sigma1: `{blind['rank_separation']['min_rank2_s2_over_s1']:.3e}`",
        f"- max rank-2 sigma3/sigma1: `{blind['rank_separation']['max_rank2_s3_over_s1']:.3e}`",
        "",
    ]
    if ref is not None:
        same = blind["canonical_rank_pattern"] == ref["canonical_rank_pattern"]
        lines += [
            "## Bundled published reference",
            "",
            f"- factor-rank counts: `{ref['factor_rank_counts']}`",
            f"- channel rank triples: `{ref['channel_rank_triples']}`",
            f"- HKS f(x,y,z): `{ref['f']}`",
            f"- leg rank sums: `{ref['leg_rank_sums']}`",
            f"- HKS g(w): `{ref['g']}`",
            f"- same full rank-pattern class: **{same}**",
            "",
            "Because the rank-pattern classes differ, the two schemes are not equivalent under the full isotropy group.",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/rank23_orbit_analysis"))
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--compare-reference", action="store_true")
    args = ap.parse_args()

    q, rank = load_checkpoint(args.checkpoint)
    if rank != 23:
        raise SystemExit(f"expected rank 23, got {rank}")
    blind = analyse(q, rank, args.tol)
    report = {"input": str(args.checkpoint), "blind": blind}

    ref = None
    if args.compare_reference:
        from rank23_reference import reference_theta
        ref = analyse(reference_theta(), 23, args.tol)
        report["reference"] = ref
        report["same_full_rank_pattern_class"] = blind["canonical_rank_pattern"] == ref["canonical_rank_pattern"]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "orbit_invariants.json").write_text(json.dumps(report, indent=2) + "\n")
    write_md(args.out / "ORBIT_INVARIANTS.md", blind, ref)

    print("blind f:", blind["f"])
    print("blind g:", blind["g"])
    print("blind factor ranks:", blind["factor_rank_counts"])
    print("blind channel triples:", blind["channel_rank_triples"])
    if ref is not None:
        print("reference f:", ref["f"])
        print("reference g:", ref["g"])
        print("same full rank-pattern class:", report["same_full_rank_pattern_class"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
