#!/usr/bin/env python3
"""Fast exact rank-pattern scan for the JKU 17,376-scheme .exp tarball.

The archive schemes use +/- 1 linear forms in a_ij, b_ij, c_ij.  This scanner
uses exact integer 3x3 ranks, so no floating tolerance enters the funnel.
"""
from __future__ import annotations

import argparse
import re
import tarfile
from collections import Counter
from itertools import permutations

TERM = re.compile(r"([+-]?)([abc])(\d)(\d)")


def form_matrix(text: str, expected: str):
    s = text.strip().replace(" ", "")
    M = [[0] * 3 for _ in range(3)]
    pos = 0
    for m in TERM.finditer(s):
        if m.start() != pos:
            raise ValueError(f"unparsed text in {s!r}: {s[pos:m.start()]!r}")
        if m.group(2) != expected:
            raise ValueError(f"expected {expected!r}, got {m.group(2)!r}")
        sign = -1 if m.group(1) == "-" else 1
        i, j = int(m.group(3)) - 1, int(m.group(4)) - 1
        M[i][j] += sign
        pos = m.end()
    if pos != len(s):
        raise ValueError(f"trailing text in {s!r}: {s[pos:]!r}")
    return M


def rank3(M):
    if not any(M[i][j] for i in range(3) for j in range(3)):
        return 0
    det = (
        M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
        - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
        + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
    )
    if det:
        return 3
    for i1 in range(3):
        for i2 in range(i1 + 1, 3):
            for j1 in range(3):
                for j2 in range(j1 + 1, 3):
                    if M[i1][j1] * M[i2][j2] - M[i1][j2] * M[i2][j1]:
                        return 2
    return 1


def parse_exp(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 23:
        raise ValueError(f"expected 23 summands, got {len(lines)}")
    pattern = []
    for line in lines:
        if not (line.startswith("(") and line.endswith(")")):
            raise ValueError(f"unexpected summand syntax: {line!r}")
        parts = line[1:-1].split(")*(")
        if len(parts) != 3:
            raise ValueError(f"expected three factors: {line!r}")
        pattern.append(
            tuple(rank3(form_matrix(part, prefix)) for part, prefix in zip(parts, "abc"))
        )
    return pattern


def canonical_pattern(pattern):
    return min(
        tuple(sorted(tuple(t[i] for i in pi) for t in pattern))
        for pi in permutations(range(3))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    args = ap.parse_args()

    # Exact target pattern from the blind rank-23 certificate.
    target_pattern = (
        [(1, 1, 1)] * 8
        + [(1, 1, 2)] * 6
        + [(2, 1, 1)] * 5
        + [(2, 2, 2)] * 4
    )
    target_canonical = canonical_pattern(target_pattern)
    target_factor = Counter({1: 46, 2: 23})
    target_sum = Counter({3: 8, 4: 11, 6: 4})

    parsed = factor = summand = full = errors = 0
    with tarfile.open(args.archive, "r:gz") as tf:
        for member in tf:
            if not (member.isfile() and member.name.endswith(".exp")):
                continue
            try:
                raw = tf.extractfile(member).read().decode("utf-8")
                pattern = parse_exp(raw)
            except Exception as exc:
                errors += 1
                print(f"PARSE ERROR {member.name}: {exc}")
                continue
            parsed += 1
            ranks = Counter(r for triple in pattern for r in triple)
            if ranks != target_factor:
                continue
            factor += 1
            if Counter(sum(t) for t in pattern) != target_sum:
                continue
            summand += 1
            if canonical_pattern(pattern) != target_canonical:
                continue
            full += 1
            print("FULL PATTERN SURVIVOR", member.name)

    print(f"parsed_rank23: {parsed}")
    print(f"parse_errors: {errors}")
    print(f"same_factor_rank_counts: {factor}")
    print(f"same_summand_rank_sum: {summand}")
    print(f"same_canonical_full_rank_pattern: {full}")


if __name__ == "__main__":
    main()
