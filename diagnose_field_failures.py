#!/usr/bin/env python3
"""Diagnose failed arithmetic recognition from saved high_precision_values.json.

This does not rerun geometry/Newton. It analyses the already-refined free coordinates,
reports individual algebraic degrees, and tries increasingly rich primitive-element
searches (individual, pair, triple, aggregate combinations) in one simple number field.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import mpmath as mp

from number_field_exact import algdep, discover_common_field, frac_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("high_precision_json", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-degree", type=int, default=16)
    ap.add_argument("--rational-den", type=int, default=10**6)
    ap.add_argument("--algdep-maxcoeff", type=int, default=10**50)
    ap.add_argument("--basis-maxcoeff", type=int, default=10**60)
    ap.add_argument("--individual-sample", type=int, default=128)
    ap.add_argument("--pair-sample", type=int, default=28)
    ap.add_argument("--triple-sample", type=int, default=12)
    ap.add_argument("--aggregate-sample", type=int, default=10)
    args = ap.parse_args()

    obj = json.loads(args.high_precision_json.read_text())
    dps = int(obj.get("dps", 220))
    mp.mp.dps = dps
    tol = mp.mpf(10) ** (-(dps - 30))
    inds = [int(x) for x in obj["free_indices"]]
    vals = [mp.mpf(x) for x in obj["values"]]

    rationals = []
    unresolved = []
    individual = []
    hist = Counter()
    for idx, x in zip(inds, vals):
        fr = Fraction(str(x)).limit_denominator(args.rational_den)
        err = abs(mp.mpf(fr.numerator)/fr.denominator - x)
        if err < tol:
            rationals.append({"index": idx, "rational": frac_str(fr), "error": mp.nstr(err, 10)})
            hist[1] += 1
        else:
            unresolved.append((idx, x))
            p = algdep(x, max_degree=args.max_degree, tol=tol, maxcoeff=args.algdep_maxcoeff)
            degree = len(p)-1 if p else None
            hist[str(degree)] += 1
            individual.append({"index": idx, "degree": degree, "minpoly": p})

    selected = None
    failure = None
    try:
        common = discover_common_field(
            unresolved,
            max_degree=args.max_degree,
            tol=tol,
            maxcoeff_algdep=args.algdep_maxcoeff,
            maxcoeff_basis=args.basis_maxcoeff,
            individual_sample=args.individual_sample,
            pair_sample=args.pair_sample,
            triple_sample=args.triple_sample,
            aggregate_sample=args.aggregate_sample,
        )
        selected = common["diagnostics"].get("selected")
        diag = common["diagnostics"]
    except RuntimeError as exc:
        diag = exc.args[1] if len(exc.args) > 1 and isinstance(exc.args[1], dict) else {}
        failure = str(exc.args[0] if exc.args else exc)

    report = {
        "input": str(args.high_precision_json),
        "dps": dps,
        "free_coordinates": len(vals),
        "rational_coordinates": len(rationals),
        "unresolved_coordinates": len(unresolved),
        "individual_degree_histogram": dict(hist),
        "individual": individual,
        "selected_common_field": selected,
        "failure": failure,
        "search_diagnostics": diag,
    }
    out = args.out or args.high_precision_json.with_name("field_failure_deep_diagnostics.json")
    out.write_text(json.dumps(report, indent=2)+"\n")

    print(f"free={len(vals)} rational={len(rationals)} unresolved={len(unresolved)}")
    print("individual degree histogram:", dict(hist))
    if selected:
        print("COMMON FIELD FOUND")
        print(" degree:", selected.get("degree"))
        print(" minpoly:", selected.get("minpoly"))
        print(" generator:", selected.get("label"))
        print(" max reconstruction error:", selected.get("max_error"))
    else:
        print("NO COMMON FIELD FOUND")
        print(" failure:", failure)
        cands = diag.get("candidates", [])
        if cands:
            # Show candidates that got furthest through the ordered coordinate list.
            best = sorted(cands, key=lambda r: (r.get("failed_index") is None, -(r.get("failed_index") or -1)), reverse=True)[:10]
            print(" candidate attempts:")
            for r in best:
                print("  ", r.get("label"), "deg", r.get("degree"), "failed", r.get("failed_index"))
    print("wrote", out)


if __name__ == "__main__":
    main()
