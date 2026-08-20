#!/usr/bin/env python3
"""Polish and structurally classify the five-seed rank-23 endpoints.

Given the output of run_5seed_replication.sh, this script:

1. locates each seed's final rank-23 checkpoint;
2. independently Gauss--Newton polishes it to a common tolerance;
3. computes factor-matrix rank fingerprints at several tolerances;
4. canonicalises the full 23-channel rank pattern under tensor-leg permutation;
5. computes HKS-style f/g summaries and a numerical sandwich-incidence WL hash;
6. groups endpoints into structural classes;
7. optionally scans the 17,376-scheme JKU .exp tar once and reports how many
   archived schemes share each unique canonical rank pattern.

The rank-pattern classification is numerical until an endpoint is exactified.
We deliberately require stability across multiple tolerances and report singular
value separation so a tolerance artefact is visible rather than silently accepted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from exactify_rank23 import load_checkpoint
from geometry_flow import gauss_newton_correct, residual_vector, unpack, unit_columns
from rankflow import mm_tensor

torch.set_default_dtype(torch.float64)

TOLS_DEFAULT = (1e-7, 1e-8, 1e-9)


def checkpoint_candidates(seed_dir: Path) -> list[Path]:
    ctrl = seed_dir / "controller_26_to_23"
    names = [
        ctrl / "final.pt",
        ctrl / "latest_exact.pt",
        ctrl / "latest_state.pt",
        ctrl / "best_delete_susceptibility.pt",
        ctrl / "best_genericity.pt",
        ctrl / "best_death_distance.pt",
    ]
    out = [p for p in names if p.exists()]
    # Beam checkpoints may contain the actual rank-23 endpoint if final.pt is absent.
    out += sorted(ctrl.glob("**/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    seen = set()
    unique = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def find_rank23_checkpoint(seed_dir: Path) -> Path:
    errors = []
    for p in checkpoint_candidates(seed_dir):
        try:
            _, rank = load_checkpoint(p)
        except Exception as exc:
            errors.append(f"{p}: {exc}")
            continue
        if rank == 23:
            return p
    raise FileNotFoundError(
        f"no rank-23 checkpoint found under {seed_dir}; "
        + ("; ".join(errors[:3]) if errors else "no .pt candidates")
    )


def mat3_rank(M: torch.Tensor, tol: float) -> tuple[int, tuple[float, float, float]]:
    s = torch.linalg.svdvals(M)
    s0 = float(s[0])
    if not math.isfinite(s0) or s0 == 0.0:
        return 0, (0.0, 0.0, 0.0)
    rel = tuple(float(x / s[0]) for x in s)
    return int(sum(x > tol for x in rel)), rel


def factors(theta: torch.Tensor, rank: int):
    U, V, W, a = unpack(theta, 3, rank)
    return unit_columns(U), unit_columns(V), unit_columns(W), a


def raw_rank_pattern(theta: torch.Tensor, rank: int, tol: float):
    U, V, W, _ = factors(theta, rank)
    pat = []
    diagnostics = []
    for r in range(rank):
        tri = []
        diag = []
        for X in (U, V, W):
            rk, rel = mat3_rank(X[:, r].reshape(3, 3), tol)
            tri.append(rk)
            diag.append(rel)
        pat.append(tuple(tri))
        diagnostics.append(diag)
    return pat, diagnostics


def canonical_pattern(pattern):
    """Canonical modulo channel permutation and S3 tensor-leg permutation."""
    pattern = list(pattern)
    return min(
        tuple(sorted((tuple(t[i] for i in pi) for t in pattern)))
        for pi in itertools.permutations(range(3))
    )


def pattern_key(canon) -> str:
    return "|".join("".join(map(str, t)) for t in canon)


def pattern_digest(canon) -> str:
    return hashlib.sha256(pattern_key(canon).encode()).hexdigest()[:16]


def fmt_poly(counter: Counter, var: str = "x") -> str:
    terms = []
    for e, n in sorted(counter.items(), reverse=True):
        if e == 0:
            mono = "1"
        elif e == 1:
            mono = var
        else:
            mono = f"{var}^{e}"
        terms.append(mono if n == 1 else f"{n}{mono}")
    return " + ".join(terms) if terms else "0"


def hks_summary(pattern):
    factor = Counter(q for t in pattern for q in t)
    triple = Counter(pattern)
    sums = [sum(t[i] for t in pattern) for i in range(3)]
    g = Counter(sums)
    f = Counter()
    for t in pattern:
        for pi in itertools.permutations(range(3)):
            f[tuple(t[i] for i in pi)] += 1
    return {
        "factor_rank_counts": {str(k): int(factor.get(k, 0)) for k in (0, 1, 2, 3)},
        "channel_rank_triples": {str(k): int(v) for k, v in sorted(triple.items())},
        "leg_rank_sums": sums,
        "g_terms": {str(k): int(v) for k, v in sorted(g.items())},
        "g": fmt_poly(g, "w"),
        "f_terms": {str(k): int(v) for k, v in sorted(f.items())},
    }


def rank_separation(pattern, diagnostics):
    r1_s2 = []
    r2_s2 = []
    r2_s3 = []
    r3_s3 = []
    for tri, diag in zip(pattern, diagnostics):
        for rk, rel in zip(tri, diag):
            if rk == 1:
                r1_s2.append(rel[1])
            elif rk == 2:
                r2_s2.append(rel[1])
                r2_s3.append(rel[2])
            elif rk == 3:
                r3_s3.append(rel[2])
    return {
        "max_rank1_sigma2_over_sigma1": max(r1_s2) if r1_s2 else None,
        "min_rank2_sigma2_over_sigma1": min(r2_s2) if r2_s2 else None,
        "max_rank2_sigma3_over_sigma1": max(r2_s3) if r2_s3 else None,
        "min_rank3_sigma3_over_sigma1": min(r3_s3) if r3_s3 else None,
    }


def product_rank_labels_numeric(theta: torch.Tensor, rank: int, tol: float):
    U, V, W, _ = factors(theta, rank)
    mats = []
    for X in (U, V, W):
        mats.append([X[:, r].reshape(3, 3) for r in range(rank)])
    nodes, _ = raw_rank_pattern(theta, rank, tol)
    edges = [[None] * rank for _ in range(rank)]
    A, B, C = mats
    for r in range(rank):
        for s in range(rank):
            edges[r][s] = (
                mat3_rank(A[r] @ B[s], tol)[0],
                mat3_rank(B[r] @ C[s], tol)[0],
                mat3_rank(C[r] @ A[s], tol)[0],
            )
    return nodes, edges


def wl_fingerprint(theta: torch.Tensor, rank: int, tol: float, rounds: int = 8) -> str:
    """Permutation-invariant directed coloured-graph hash.

    This numerical version is diagnostic only. It is stable/useful after polishing,
    but exact inequivalence claims should use the exact certificate implementation.
    """
    nodes, edges = product_rank_labels_numeric(theta, rank, tol)
    colors = [repr(x) for x in nodes]
    for _ in range(rounds):
        raw = []
        for i in range(rank):
            out = sorted((edges[i][j], colors[j]) for j in range(rank))
            inc = sorted((edges[j][i], colors[j]) for j in range(rank))
            raw.append(hashlib.sha256(repr((colors[i], out, inc)).encode()).hexdigest())
        uniq = {v: k for k, v in enumerate(sorted(set(raw)))}
        new = [str(uniq[v]) for v in raw]
        if new == colors:
            break
        colors = new
    edge_hist = Counter(x for row in edges for x in row)
    return hashlib.sha256(
        repr((sorted(Counter(colors).items()), sorted(edge_hist.items()))).encode()
    ).hexdigest()


def parity(pi):
    return sum(pi[i] > pi[j] for i in range(3) for j in range(i + 1, 3)) & 1


def transformed_for_leg_action(theta: torch.Tensor, rank: int, pi):
    """Apply only tensor-leg permutation/transposition to numerical factors.

    Used solely to canonicalise the diagnostic WL fingerprint over S3.
    """
    U, V, W, a = factors(theta, rank)
    legs = [U, V, W]
    new = [legs[i].clone() for i in pi]
    if parity(pi):
        trans = []
        for X in new:
            Y = torch.empty_like(X)
            for r in range(rank):
                Y[:, r] = X[:, r].reshape(3, 3).T.reshape(-1)
            trans.append(Y)
        new = trans
    from geometry_flow import pack
    return pack(new[0], new[1], new[2], a.clone())


def canonical_wl(theta: torch.Tensor, rank: int, tol: float) -> str:
    vals = []
    for pi in itertools.permutations(range(3)):
        q = transformed_for_leg_action(theta, rank, pi)
        vals.append(wl_fingerprint(q, rank, tol))
    return min(vals)


def polish(q0: torch.Tensor, rank: int, tol: float, iters: int, rcond: float):
    T = mm_tensor(3)
    before = float(residual_vector(q0, T, 3, rank).norm())
    q, info = gauss_newton_correct(
        q0, T, 3, rank, tol=tol, max_iters=iters, rcond=rcond
    )
    after = float(residual_vector(q, T, 3, rank).norm())
    return q, before, after, info


# ----- fast exact .exp tar scan for optional JKU comparison -----

TERM = re.compile(r"([+-]?)([abc])(\d)(\d)")


def exp_form_matrix(text: str, expected: str):
    s = text.strip().replace(" ", "")
    M = [[0] * 3 for _ in range(3)]
    pos = 0
    for m in TERM.finditer(s):
        if m.start() != pos:
            raise ValueError(f"unparsed text {s[pos:m.start()]!r}")
        if m.group(2) != expected:
            raise ValueError(f"expected {expected}, got {m.group(2)}")
        sign = -1 if m.group(1) == "-" else 1
        M[int(m.group(3))-1][int(m.group(4))-1] += sign
        pos = m.end()
    if pos != len(s):
        raise ValueError(f"trailing text {s[pos:]!r}")
    return M


def rank3_int(M):
    if not any(x for row in M for x in row):
        return 0
    a,b,c = M[0]; d,e,f = M[1]; g,h,i = M[2]
    det = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
    if det:
        return 3
    minors = (
        a*e-b*d, a*f-c*d, b*f-c*e,
        a*h-b*g, a*i-c*g, b*i-c*h,
        d*h-e*g, d*i-f*g, e*i-f*h,
    )
    return 2 if any(minors) else 1


def parse_exp_pattern(text: str):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) != 23:
        raise ValueError(f"expected 23 summands, got {len(lines)}")
    pat = []
    for line in lines:
        if not (line.startswith("(") and line.endswith(")")):
            raise ValueError("unexpected summand syntax")
        parts = line[1:-1].split(")*(")
        if len(parts) != 3:
            raise ValueError("expected three factors")
        pat.append(tuple(rank3_int(exp_form_matrix(p, c)) for p, c in zip(parts, "abc")))
    return pat


def scan_jku_patterns(archive: Path, targets: dict[str, tuple]):
    counts = Counter()
    parsed = errors = 0
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not (member.isfile() and member.name.endswith(".exp")):
                continue
            try:
                raw = tf.extractfile(member).read().decode("utf-8")
                canon = canonical_pattern(parse_exp_pattern(raw))
            except Exception:
                errors += 1
                continue
            parsed += 1
            for cls, target in targets.items():
                if canon == target:
                    counts[cls] += 1
    return parsed, errors, counts


def fmt_e(x):
    if x is None:
        return "—"
    return f"{x:.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root", type=Path, nargs="?", default=Path("results/replication_5seeds"),
        help="five-seed replication root"
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--refine-tol", type=float, default=2e-14)
    ap.add_argument("--refine-iters", type=int, default=30)
    ap.add_argument("--rcond", type=float, default=1e-10)
    ap.add_argument(
        "--rank-tols", type=float, nargs="+", default=list(TOLS_DEFAULT),
        help="rank tolerances; structural class must be stable across all"
    )
    ap.add_argument("--jku-tar", type=Path, default=None)
    args = ap.parse_args()

    root = args.root
    out = args.out or (root / "endpoint_analysis")
    out.mkdir(parents=True, exist_ok=True)
    polished_dir = out / "polished"
    polished_dir.mkdir(exist_ok=True)

    rows = []
    full_reports = {}
    class_canons = {}
    next_class = 1
    canon_to_class = {}

    seed_dirs = sorted(
        [p for p in root.glob("seed_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_", 1)[1])
    )
    if not seed_dirs:
        raise SystemExit(f"no seed_* directories found under {root}")

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_", 1)[1])
        try:
            ckpt = find_rank23_checkpoint(seed_dir)
            q0, rank = load_checkpoint(ckpt)
            if rank != 23:
                raise ValueError(f"expected rank 23, got {rank}")
            q, before, after, info = polish(
                q0, rank, args.refine_tol, args.refine_iters, args.rcond
            )
        except Exception as exc:
            rows.append({
                "seed": seed, "status": "ERROR", "error": repr(exc),
            })
            continue

        U, V, W, a = unpack(q, 3, rank)
        polished_path = polished_dir / f"seed_{seed}_rank23_refined.pt"
        torch.save(
            {
                "theta": q,
                "rank": rank,
                "U": U, "V": V, "W": W, "a": a,
                "source_checkpoint": str(ckpt),
            },
            polished_path,
        )

        tol_reports = {}
        canon_keys = []
        wl_hashes = []
        for tol in args.rank_tols:
            pat, diag = raw_rank_pattern(q, rank, tol)
            canon = canonical_pattern(pat)
            canon_keys.append(pattern_key(canon))
            wl_hashes.append(canonical_wl(q, rank, tol))
            tol_reports[str(tol)] = {
                "pattern": [list(x) for x in pat],
                "canonical_pattern": [list(x) for x in canon],
                "canonical_pattern_digest": pattern_digest(canon),
                "rank_separation": rank_separation(pat, diag),
                "hks": hks_summary(pat),
                "wl_hash": wl_hashes[-1],
            }

        stable_pattern = len(set(canon_keys)) == 1
        stable_wl = len(set(wl_hashes)) == 1
        central_tol = min(args.rank_tols, key=lambda x: abs(math.log10(x) + 8))
        central = tol_reports[str(central_tol)]
        canon = tuple(tuple(x) for x in central["canonical_pattern"])
        if stable_pattern:
            if canon not in canon_to_class:
                cname = f"C{next_class}"
                next_class += 1
                canon_to_class[canon] = cname
                class_canons[cname] = canon
            cls = canon_to_class[canon]
        else:
            cls = "UNSTABLE"

        max_amp = float(a.abs().max())
        sep = central["rank_separation"]
        row = {
            "seed": seed,
            "status": "OK",
            "source": str(ckpt),
            "polished": str(polished_path),
            "input_residual": before,
            "polished_residual": after,
            "refine_converged": bool(info.converged),
            "refine_iterations": int(info.iterations),
            "max_abs_amplitude": max_amp,
            "class": cls,
            "stable_rank_pattern": stable_pattern,
            "stable_wl": stable_wl,
            "pattern_digest": central["canonical_pattern_digest"],
            "wl_hash": central["wl_hash"][:16],
            "factor_rank_counts": central["hks"]["factor_rank_counts"],
            "channel_rank_triples": central["hks"]["channel_rank_triples"],
            "leg_rank_sums": central["hks"]["leg_rank_sums"],
            "max_rank1_s2_s1": sep["max_rank1_sigma2_over_sigma1"],
            "min_rank2_s2_s1": sep["min_rank2_sigma2_over_sigma1"],
            "max_rank2_s3_s1": sep["max_rank2_sigma3_over_sigma1"],
        }
        rows.append(row)
        full_reports[str(seed)] = {
            "row": row,
            "rank_tolerances": tol_reports,
        }

    # Optional exact archive comparison for each *numerically stable* pattern class.
    jku = None
    if args.jku_tar is not None:
        targets = {c: pat for c, pat in class_canons.items()}
        parsed, errors, counts = scan_jku_patterns(args.jku_tar, targets)
        jku = {
            "archive": str(args.jku_tar),
            "parsed_rank23": parsed,
            "parse_errors": errors,
            "canonical_pattern_matches": dict(counts),
        }
        for row in rows:
            if row.get("class") in targets:
                row["jku_same_canonical_pattern"] = int(counts.get(row["class"], 0))

    report = {
        "root": str(root),
        "rank_tolerances": args.rank_tols,
        "rows": rows,
        "classes": {
            cls: {
                "canonical_pattern": [list(x) for x in pat],
                "pattern_digest": pattern_digest(pat),
                "seeds": [r["seed"] for r in rows if r.get("class") == cls],
            }
            for cls, pat in class_canons.items()
        },
        "jku": jku,
        "endpoint_details": full_reports,
    }
    (out / "endpoint_analysis.json").write_text(json.dumps(report, indent=2) + "\n")

    # CSV
    csv_fields = [
        "seed","status","class","input_residual","polished_residual",
        "refine_converged","refine_iterations","max_abs_amplitude",
        "stable_rank_pattern","stable_wl","pattern_digest","wl_hash",
        "max_rank1_s2_s1","min_rank2_s2_s1","max_rank2_s3_s1",
        "jku_same_canonical_pattern","source","polished","error",
    ]
    with (out / "endpoint_analysis.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("status") == "OK"]
    classes = defaultdict(list)
    for r in ok:
        classes[r["class"]].append(r["seed"])

    lines = [
        "# Five-seed rank-23 endpoint analysis",
        "",
        f"- endpoints analysed: **{len(ok)}/{len(rows)}**",
        f"- stable canonical rank-pattern classes: **{len([c for c in classes if c != 'UNSTABLE'])}**",
        f"- all endpoints stable across tolerances: **{all(r['stable_rank_pattern'] for r in ok)}**",
        "",
        "| seed | class | input residual | polished residual | max |a| | stable pattern? | stable WL? | pattern hash |",
        "|---:|:---:|---:|---:|---:|:---:|:---:|:---|",
    ]
    for r in rows:
        if r.get("status") != "OK":
            lines.append(f"| {r['seed']} | ERROR | — | — | — | — | — | `{r.get('error','')}` |")
        else:
            lines.append(
                f"| {r['seed']} | {r['class']} | {r['input_residual']:.3e} | "
                f"{r['polished_residual']:.3e} | {r['max_abs_amplitude']:.3f} | "
                f"{'YES' if r['stable_rank_pattern'] else 'NO'} | "
                f"{'YES' if r['stable_wl'] else 'NO'} | `{r['pattern_digest']}` |"
            )

    lines += ["", "## Structural classes", ""]
    for cls, seeds in sorted(classes.items()):
        if cls == "UNSTABLE":
            lines.append(f"- **UNSTABLE**: seeds {seeds}")
            continue
        rr = next(r for r in ok if r["class"] == cls)
        lines.append(
            f"- **{cls}**: seeds {seeds}; factor ranks `{rr['factor_rank_counts']}`; "
            f"channel triples `{rr['channel_rank_triples']}`"
        )
        if jku is not None:
            lines[-1] += f"; JKU same canonical pattern: **{rr.get('jku_same_canonical_pattern', 0)}**"

    if len([c for c in classes if c != "UNSTABLE"]) == 1 and ok:
        lines += [
            "",
            "All numerically stable endpoints share the same canonical factor-rank pattern.",
            "This is evidence that the frozen controller repeatedly reaches the same structural rank-23 class,",
            "but it is not by itself an exact orbit-equivalence proof between the numerical endpoints.",
        ]
    elif len([c for c in classes if c != "UNSTABLE"]) > 1:
        lines += [
            "",
            "The frozen controller reached multiple distinct canonical factor-rank patterns.",
            "Once exactified, differing stable patterns would prove the endpoints lie in inequivalent isotropy orbits.",
        ]

    if jku is not None:
        lines += [
            "",
            "## JKU archive comparison",
            "",
            f"- parsed `.exp` schemes: **{jku['parsed_rank23']}**",
            f"- parse errors: **{jku['parse_errors']}**",
            "",
            "Archive counts use exact integer factor ranks. Endpoint classes are still numerical until exactified,",
            "so a zero match is strong classification evidence but should be phrased conservatively until the endpoint is exactified.",
        ]

    (out / "ENDPOINT_ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
