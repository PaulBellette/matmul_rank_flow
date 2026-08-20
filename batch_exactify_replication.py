#!/usr/bin/env python3
"""Batch exactify and classify replicated rank-23 endpoints.

This is a downstream analysis tool. It does not alter the discovery controller.
For each seed it:
  1. uses the polished rank-23 endpoint from endpoint_analysis;
  2. finds a well-conditioned incidence isotropy gauge;
  3. chooses a structural-zero threshold by an explicit residual-preservation test;
  4. runs sparse-family exactification;
  5. verifies the resulting 729 Brent identities exactly;
  6. computes the exact canonical factor-rank pattern;
  7. optionally scans the 17,376-scheme JKU tar once for exact pattern matches.

Failures are per-seed and fail-soft: one arithmetic field / exactification failure does
not abort the remaining seeds.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import sympy as sp
import torch

from exactify_rank23 import canonical_channel_gauge, load_checkpoint, tensor_from_canonical
from rankflow import mm_tensor
from scan_jku_tar import parse_exp, canonical_pattern as canonical_exp_pattern
from number_field_exact import SimpleNumberField
from fractions import Fraction


def run(cmd, cwd: Path, log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return p.returncode


def choose_zero_threshold(checkpoint: Path, rank: int = 23):
    q, rr = load_checkpoint(checkpoint)
    if rr != rank:
        raise ValueError(f"expected rank {rank}, got {rr}")
    U, V, W, c, pivots = canonical_channel_gauge(q, 3, rank)
    T = mm_tensor(3)
    candidates = [1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    rows = []
    good = []
    for t in candidates:
        A = [U.clone(), V.clone(), W.clone()]
        nzero = 0
        for X in A:
            mask = X.abs() < t
            nzero += int(mask.sum())
            X[mask] = 0.0
        # Reinstate pivots exactly to avoid a threshold ever touching them.
        for r, pp in enumerate(pivots):
            for leg, i in enumerate(pp):
                A[leg][i, r] = 1.0
        res = float((tensor_from_canonical(A[0], A[1], A[2], c) - T).norm())
        row = {"threshold": t, "factor_zeros": nzero, "zeroed_tensor_residual": res}
        rows.append(row)
        if res < 1e-8:
            good.append(row)
    if not good:
        raise RuntimeError("no residual-preserving structural-zero threshold found")
    # Largest threshold that still leaves the exact tensor unchanged to a very tight numerical scale.
    chosen = good[-1]
    return chosen["threshold"], rows


def parse_expr_grid(rows):
    return [[sp.sympify(x) for x in row] for row in rows]


def exact_rank3(M):
    M = sp.Matrix(M)
    # rank() is exact over algebraic expressions but simplification of all minors is more predictable for 3x3.
    d = sp.simplify(M.det())
    if d != 0:
        return 3
    for I in itertools.combinations(range(3), 2):
        for J in itertools.combinations(range(3), 2):
            if sp.simplify(M.extract(I, J).det()) != 0:
                return 2
    return 0 if all(sp.simplify(x) == 0 for x in M) else 1



def _parse_frac(s):
    return Fraction(str(s))


def _parse_basis_grid(rows, field):
    return [[field.elt([_parse_frac(q) for q in entry]) for entry in row] for row in rows]


def certificate_pattern(cert_path: Path):
    cert = json.loads(cert_path.read_text())
    if not cert.get("exact_identity"):
        raise ValueError("certificate does not verify exactly")
    rank = int(cert["rank"])

    if cert.get("number_field") and cert.get("U_power_basis"):
        nf = cert["number_field"]
        mp = tuple(Fraction(x) for x in nf["minimal_polynomial_coefficients_ascending"])
        field = SimpleNumberField(mp)
        U = _parse_basis_grid(cert["U_power_basis"], field)
        V = _parse_basis_grid(cert["V_power_basis"], field)
        W = _parse_basis_grid(cert["W_power_basis"], field)
        pattern = []
        for r in range(rank):
            tri = []
            for X in (U, V, W):
                M = [[X[3*i+j][r] for j in range(3)] for i in range(3)]
                tri.append(field.rank3(M))
            pattern.append(tuple(tri))
    else:
        # Backward compatibility with the first quadratic SymPy certificate.
        U = parse_expr_grid(cert["U"])
        V = parse_expr_grid(cert["V"])
        W = parse_expr_grid(cert["W"])
        pattern = []
        for r in range(rank):
            tri = []
            for X in (U, V, W):
                M = [[X[3*i+j][r] for j in range(3)] for i in range(3)]
                tri.append(exact_rank3(M))
            pattern.append(tuple(tri))
    canon = min(
        tuple(sorted(tuple(t[i] for i in pi) for t in pattern))
        for pi in itertools.permutations(range(3))
    )
    return pattern, canon

def pattern_summary(pattern):
    factor = Counter(r for t in pattern for r in t)
    triples = Counter(pattern)
    return {
        "factor_rank_counts": {str(k): factor.get(k, 0) for k in range(4)},
        "channel_rank_triples": {str(k): v for k, v in sorted(triples.items())},
        "summand_rank_sums": {str(k): v for k, v in sorted(Counter(sum(t) for t in pattern).items())},
    }


def scan_jku_tar(archive: Path, targets: dict[str, tuple]):
    counts = Counter()
    parsed = errors = 0
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not (member.isfile() and member.name.endswith(".exp")):
                continue
            try:
                raw = tf.extractfile(member).read().decode("utf-8")
                canon = canonical_exp_pattern(parse_exp(raw))
            except Exception:
                errors += 1
                continue
            parsed += 1
            for key, target in targets.items():
                if canon == target:
                    counts[key] += 1
    return parsed, errors, counts


def seed_checkpoints(root: Path):
    polished = root / "endpoint_analysis" / "polished"
    out = []
    for p in sorted(polished.glob("seed_*_rank23_refined.pt")):
        try:
            seed = int(p.name.split("_")[1])
        except Exception:
            continue
        out.append((seed, p))
    if out:
        return out
    # fallback: find final checkpoints if endpoint analysis was run elsewhere
    for sd in sorted(root.glob("seed_*")):
        try:
            seed = int(sd.name.split("_", 1)[1])
        except Exception:
            continue
        p = sd / "controller_26_to_23" / "final.pt"
        if p.exists():
            out.append((seed, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", type=Path, default=Path("results/replication_5seeds"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--jku-tar", type=Path, default=None)
    ap.add_argument("--dps", type=int, default=130)
    ap.add_argument("--rcond", type=float, default=1e-10)
    ap.add_argument("--condition-limit", type=float, default=20.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-field-degree", type=int, default=10)
    args = ap.parse_args()

    root = args.root.resolve()
    repo = Path(__file__).resolve().parent
    out = (args.out or root / "exact_endpoints").resolve()
    out.mkdir(parents=True, exist_ok=True)

    seeds = seed_checkpoints(root)
    if not seeds:
        raise SystemExit(f"no rank-23 seed checkpoints found under {root}")

    rows = []
    exact_canons = {}
    for seed, checkpoint in seeds:
        t0 = time.time()
        sd = out / f"seed_{seed}"
        incidence = sd / "incidence"
        exact = sd / "exact"
        sd.mkdir(parents=True, exist_ok=True)
        row = {"seed": seed, "checkpoint": str(checkpoint), "status": "started"}
        print(f"\n=== seed {seed} ===", flush=True)

        try:
            sparse_pt = incidence / "rank23_incidence_sparse.pt"
            if args.force or not sparse_pt.exists():
                rc = run([
                    sys.executable, str(repo / "isotropy_incidence_gauge.py"), str(checkpoint),
                    "--out", str(incidence), "--condition-limit", str(args.condition_limit),
                ], repo, sd / "incidence.log")
                if rc:
                    raise RuntimeError(f"incidence gauge failed rc={rc}; see {sd/'incidence.log'}")

            threshold, threshold_rows = choose_zero_threshold(sparse_pt)
            row["zero_threshold"] = threshold
            (sd / "zero_threshold_scan.json").write_text(json.dumps(threshold_rows, indent=2) + "\n")
            print(f"seed {seed}: structural zero threshold {threshold:g}", flush=True)

            cert = exact / "rank23_exact.json"
            if args.force or not cert.exists():
                rc = run([
                    sys.executable, str(repo / "sparse_family_exactify.py"), str(sparse_pt),
                    "--out", str(exact), "--zero-threshold", str(threshold),
                    "--dps", str(args.dps), "--rcond", str(args.rcond), "--max-field-degree", str(args.max_field_degree),
                ], repo, sd / "exactify.log")
                if rc:
                    raise RuntimeError(f"sparse exactification failed rc={rc}; see {sd/'exactify.log'}")

            cert_obj = json.loads(cert.read_text())
            if not cert_obj.get("exact_identity"):
                raise RuntimeError("exactification produced a non-verifying certificate")
            pattern, canon = certificate_pattern(cert)
            summ = pattern_summary(pattern)
            exact_canons[str(seed)] = canon
            report = json.loads((exact / "sparse_exact_report.json").read_text())
            row.update({
                "status": "exact",
                "field": cert_obj.get("field"),
                "rational_coefficients": cert_obj.get("coefficient_counts", {}).get("rational"),
                "algebraic_coefficients": cert_obj.get("coefficient_counts", {}).get("algebraic", cert_obj.get("coefficient_counts", {}).get("quadratic")),
                "field_degree": cert_obj.get("field_degree", cert_obj.get("number_field", {}).get("degree")),
                "family_nullity": report.get("initial_family_nullity"),
                "family_move_l2": report.get("family_move_l2"),
                "family_move_max": report.get("family_move_max"),
                "factor_rank_counts": summ["factor_rank_counts"],
                "channel_rank_triples": summ["channel_rank_triples"],
                "canonical_pattern": [list(x) for x in canon],
                "exact_certificate": str(cert),
            })
            print(f"seed {seed}: EXACT {row['field']} ranks={summ['factor_rank_counts']}", flush=True)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)
            print(f"seed {seed}: FAILED: {exc}", flush=True)
        row["seconds"] = time.time() - t0
        rows.append(row)
        (out / "batch_progress.json").write_text(json.dumps(rows, indent=2) + "\n")

    jku = None
    if args.jku_tar and exact_canons:
        print("\n=== scanning JKU archive once for exact endpoint classes ===", flush=True)
        parsed, errors, counts = scan_jku_tar(args.jku_tar, exact_canons)
        jku = {"archive": str(args.jku_tar), "parsed": parsed, "errors": errors, "matches": dict(counts)}
        for row in rows:
            if row["status"] == "exact":
                row["jku_same_canonical_pattern"] = int(counts.get(str(row["seed"]), 0))

    result = {"rows": rows, "jku": jku}
    (out / "batch_exactification.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = [
        "seed", "status", "field", "family_nullity", "family_move_l2", "family_move_max",
        "zero_threshold", "rational_coefficients", "algebraic_coefficients", "field_degree",
        "jku_same_canonical_pattern", "seconds", "checkpoint", "exact_certificate", "error"
    ]
    with (out / "batch_exactification.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    lines = [
        "# Five-seed exact endpoint classification", "",
        f"- endpoints attempted: **{len(rows)}**",
        f"- exact symbolic certificates: **{sum(r['status']=='exact' for r in rows)}/{len(rows)}**",
        "",
        "| seed | status | exact field | family nullity | family move L2 | exact factor ranks | JKU same full pattern |",
        "|---:|:---:|:---|---:|---:|:---|---:|",
    ]
    for r in rows:
        if r["status"] == "exact":
            fr = r["factor_rank_counts"]
            lines.append(
                f"| {r['seed']} | EXACT | `{r['field']}` | {r.get('family_nullity','')} | "
                f"{r.get('family_move_l2',0):.3e} | `{fr}` | {r.get('jku_same_canonical_pattern','—')} |"
            )
        else:
            lines.append(f"| {r['seed']} | FAILED | — | — | — | — | — |")
    if jku:
        lines += ["", "## JKU scan", "", f"- parsed: **{jku['parsed']}**", f"- errors: **{jku['errors']}**"]
    lines += [
        "", "## Interpretation", "",
        "A zero JKU full-pattern match is a rigorous inequivalence certificate for an exactified endpoint,",
        "because factor-matrix ranks and their per-channel pattern are preserved by channel permutation, CP scaling,",
        "the GL(3)^3 matrix-multiplication isotropy action, and tensor-leg permutation.",
        "", "A failed exactification is not evidence against the endpoint; it means this particular sparse-family",
        "recogniser was insufficient and the seed should be analysed separately; field-recognition diagnostics are saved per seed.",
    ]
    (out / "EXACT_ENDPOINTS.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
