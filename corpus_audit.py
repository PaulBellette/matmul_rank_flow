"""Audit local matrix-multiplication corpora before equivalence claims.

This walks every file and explains why it was parsed, rejected, or ignored.  It
shares the exact parsers with ``rank23_equivalence_search.py`` so audit counts
and search counts cannot silently disagree.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rank23_equivalence_search import scan_corpus_detailed, write_corpus_audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, action="append", required=True,
                    help="corpus root; repeat for multiple repositories")
    ap.add_argument("--out", type=Path, default=Path("results/blind_rank23/corpus_audit"))
    ap.add_argument("--samples", type=int, default=12,
                    help="sample paths retained for each status in JSON")
    args = ap.parse_args()

    reports = []
    records = []
    for root in args.corpus:
        _, report, recs = scan_corpus_detailed(root)
        reports.append(report)
        records.append(recs)

    write_corpus_audit(args.out, reports, records, sample_limit=args.samples)

    total = sum(r["parsed_rank23"] for r in reports)
    print("rank-23 corpus audit")
    for r in reports:
        print(f"{r['root']}: files_seen={r['files_seen']} candidate_files={r['candidate_files']} "
              f"plausible_by_name={r['plausible_rank23_by_name']} parsed_rank23={r['parsed_rank23']}")
        for status, count in r["status_counts"].items():
            if status != "unsupported_extension" or count < 10:
                print(f"  {status}: {count}")
        gaps = {k: v for k, v in r["status_counts"].items()
                if k in {"plausible_unsupported_extension", "parse_error", "loader_rejected",
                         "unsupported_reduced_json", "unsupported_json_schema", "json_decode_error"}}
        if gaps:
            print("  COVERAGE_GAPS:", gaps)
    print("total parsed_rank23:", total)
    print("wrote", args.out / "CORPUS_AUDIT.md")
    print("wrote", args.out / "corpus_files.csv")


if __name__ == "__main__":
    main()
