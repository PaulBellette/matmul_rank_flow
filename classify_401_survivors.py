#!/usr/bin/env python3
"""Exact stronger classification for the seven JKU schemes sharing seed 401's full rank pattern.

This is deliberately a *proof by invariant* tool. It loads seed 401's exact certificate,
recovers the exact cyclic A,B,C factors, scans the 17,376-scheme JKU .exp tar for schemes
with the same canonical full factor-rank pattern, then compares a stronger exact coloured
sandwich-incidence graph.

For channels r,s the directed edge label is
    (rank(A_r B_s), rank(B_r C_s), rank(C_r A_s)).
These labels are preserved by the standard GL(3)^3 sandwich action. Tensor-leg permutations
are handled explicitly (odd permutations include transpose, as for the trace tensor).

If no coloured graph isomorphism exists for any of the six tensor-leg actions, the candidate
is rigorously inequivalent to the target. A surviving graph-isomorphic candidate is only a
candidate for equivalence and needs the final normal-form/GL reconstruction stage.
"""
from __future__ import annotations

import argparse
import itertools
import json
import tarfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

import networkx as nx

from number_field_exact import SimpleNumberField
from scan_jku_tar import form_matrix, rank3 as rank3_int, canonical_pattern, parse_exp


def _frac(x):
    return Fraction(str(x))


def _parse_field_grid(rows, field: SimpleNumberField):
    return [[field.elt([_frac(q) for q in entry]) for entry in row] for row in rows]


def load_exact_cyclic_certificate(path: Path):
    cert = json.loads(path.read_text())
    if not cert.get("exact_identity"):
        raise ValueError("target certificate is not marked exact")
    if not cert.get("number_field") or not cert.get("U_power_basis"):
        raise ValueError("this tool expects the generic power-basis exact certificate format")

    nf = cert["number_field"]
    minpoly = tuple(_frac(x) for x in nf["minimal_polynomial_coefficients_ascending"])
    field = SimpleNumberField(minpoly)
    U = _parse_field_grid(cert["U_power_basis"], field)
    V = _parse_field_grid(cert["V_power_basis"], field)
    Wout = _parse_field_grid(cert["W_power_basis"], field)
    c = [field.elt([_frac(q) for q in entry]) for entry in cert["c_power_basis"]]
    rank = int(cert["rank"])

    def col_matrix(X, r):
        return [[X[3*i+j][r] for j in range(3)] for i in range(3)]

    A = [col_matrix(U, r) for r in range(rank)]
    B = [col_matrix(V, r) for r in range(rank)]
    # Certificate W is output-dual; cyclic trace third factor is transpose(W).
    C = []
    for r in range(rank):
        M = col_matrix(Wout, r)
        C.append([[M[j][i] for j in range(3)] for i in range(3)])

    # Bake nonzero channel amplitudes into A. This is unnecessary for ranks but makes
    # the internal representation faithful to the trilinear summands.
    for r in range(rank):
        if field.is_zero(c[r]):
            raise ValueError(f"zero channel coefficient at {r}")
        A[r] = [[field.mul(c[r], x) for x in row] for row in A[r]]
    return field, (A, B, C)


def matmul_field(X, Y, field):
    Z = [[field.zero() for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            s = field.zero()
            for k in range(3):
                s = field.add(s, field.mul(X[i][k], Y[k][j]))
            Z[i][j] = s
    return Z


def transpose(M):
    return [[M[j][i] for j in range(3)] for i in range(3)]


def parity(pi):
    return sum(pi[i] > pi[j] for i in range(3) for j in range(i+1, 3)) & 1


def field_rank_pattern(factors, field):
    A, B, C = factors
    return [(field.rank3(A[r]), field.rank3(B[r]), field.rank3(C[r])) for r in range(len(A))]


def field_leg_action(factors, pi):
    legs = [factors[i] for i in pi]
    if parity(pi):
        legs = [[transpose(M) for M in leg] for leg in legs]
    return tuple(legs)


def field_graph(factors, field):
    A, B, C = factors
    n = len(A)
    G = nx.DiGraph()
    for r, lab in enumerate(field_rank_pattern(factors, field)):
        G.add_node(r, label=lab)
    for r in range(n):
        for s in range(n):
            lab = (
                field.rank3(matmul_field(A[r], B[s], field)),
                field.rank3(matmul_field(B[r], C[s], field)),
                field.rank3(matmul_field(C[r], A[s], field)),
            )
            G.add_edge(r, s, label=lab)
    return G


def matmul_int(X, Y):
    return [[sum(X[i][k]*Y[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def int_leg_action(factors, pi):
    legs = [factors[i] for i in pi]
    if parity(pi):
        legs = [[transpose(M) for M in leg] for leg in legs]
    return tuple(legs)


def int_rank_pattern(factors):
    A, B, C = factors
    return [(rank3_int(A[r]), rank3_int(B[r]), rank3_int(C[r])) for r in range(len(A))]


def int_graph(factors):
    A, B, C = factors
    n = len(A)
    G = nx.DiGraph()
    for r, lab in enumerate(int_rank_pattern(factors)):
        G.add_node(r, label=lab)
    for r in range(n):
        for s in range(n):
            G.add_edge(r, s, label=(
                rank3_int(matmul_int(A[r], B[s])),
                rank3_int(matmul_int(B[r], C[s])),
                rank3_int(matmul_int(C[r], A[s])),
            ))
    return G


def parse_exp_factors(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 23:
        raise ValueError(f"expected 23 summands, got {len(lines)}")
    A=[];B=[];C=[]
    for line in lines:
        if not (line.startswith("(") and line.endswith(")")):
            raise ValueError("unexpected .exp syntax")
        parts = line[1:-1].split(")*(")
        if len(parts) != 3:
            raise ValueError("expected three factors")
        A.append(form_matrix(parts[0], "a"))
        B.append(form_matrix(parts[1], "b"))
        C.append(form_matrix(parts[2], "c"))
    return (A,B,C)


def edge_hist(G):
    return Counter(data["label"] for _,_,data in G.edges(data=True))


def node_hist(G):
    return Counter(data["label"] for _,data in G.nodes(data=True))


def graph_isomorphic(Gt, Gc):
    nm = nx.algorithms.isomorphism.categorical_node_match("label", None)
    em = nx.algorithms.isomorphism.categorical_edge_match("label", None)
    gm = nx.algorithms.isomorphism.DiGraphMatcher(Gt, Gc, node_match=nm, edge_match=em)
    return gm.is_isomorphic(), (next(gm.isomorphisms_iter()) if gm.is_isomorphic() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("certificate", type=Path, help="seed 401 exact/rank23_exact.json")
    ap.add_argument("archive", type=Path, help="schemes-exp.tgz")
    ap.add_argument("--out", type=Path, default=Path("results/replication_5seeds/exact_endpoints/seed_401/jku_strong"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    field, target_factors = load_exact_cyclic_certificate(args.certificate)
    target_pattern = field_rank_pattern(target_factors, field)
    target_canon = canonical_pattern(target_pattern)
    target_graph = field_graph(target_factors, field)
    target_nodes = node_hist(target_graph)
    target_edges = edge_hist(target_graph)

    full_pattern = []
    results = []
    parsed = errors = 0
    with tarfile.open(args.archive, "r:gz") as tf:
        for member in tf:
            if not (member.isfile() and member.name.endswith(".exp")):
                continue
            try:
                raw = tf.extractfile(member).read().decode("utf-8")
                pat = parse_exp(raw)
            except Exception as exc:
                errors += 1
                continue
            parsed += 1
            if canonical_pattern(pat) != target_canon:
                continue
            full_pattern.append(member.name)
            factors = parse_exp_factors(raw)

            survived_hist = False
            survived_graph = False
            graph_pi = None
            mapping = None
            action_rows = []
            for pi in itertools.permutations(range(3)):
                cand = int_leg_action(factors, pi)
                Gc = int_graph(cand)
                hist_match = node_hist(Gc) == target_nodes and edge_hist(Gc) == target_edges
                row = {"leg_permutation": list(pi), "hist_match": hist_match}
                if hist_match:
                    survived_hist = True
                    iso, mp = graph_isomorphic(target_graph, Gc)
                    row["graph_isomorphic"] = bool(iso)
                    if iso:
                        survived_graph = True
                        graph_pi = pi
                        mapping = {str(k): int(v) for k,v in mp.items()} if mp else None
                        action_rows.append(row)
                        break
                else:
                    row["graph_isomorphic"] = False
                action_rows.append(row)
            results.append({
                "member": member.name,
                "survived_edge_histogram": survived_hist,
                "survived_exact_graph_isomorphism": survived_graph,
                "leg_permutation": list(graph_pi) if graph_pi else None,
                "channel_mapping": mapping,
                "actions": action_rows,
            })

    survivors = [r for r in results if r["survived_exact_graph_isomorphism"]]
    hist_survivors = [r for r in results if r["survived_edge_histogram"]]
    report = {
        "target_certificate": str(args.certificate),
        "archive": str(args.archive),
        "archive_parsed": parsed,
        "archive_parse_errors": errors,
        "same_canonical_full_rank_pattern": len(full_pattern),
        "same_exact_sandwich_edge_histogram": len(hist_survivors),
        "same_exact_coloured_incidence_graph": len(survivors),
        "full_pattern_members": full_pattern,
        "results": results,
    }
    (args.out / "seed401_jku_strong.json").write_text(json.dumps(report, indent=2)+"\n")

    lines = [
        "# Seed 401: exact JKU survivor classification", "",
        f"- archive schemes parsed: **{parsed}**",
        f"- parse errors: **{errors}**",
        f"- same canonical full factor-rank pattern: **{len(full_pattern)}**",
        f"- same exact sandwich edge-label histogram: **{len(hist_survivors)}**",
        f"- same exact coloured sandwich-incidence graph: **{len(survivors)}**", "",
    ]
    if not survivors:
        lines += [
            "**Result:** all full-pattern survivors are rigorously inequivalent to seed 401 under the standard matrix-multiplication isotropy action.",
            "A genuine equivalence would preserve the labelled sandwich-incidence graph (up to channel and tensor-leg permutations), so absence of a graph isomorphism is a certificate of inequivalence.", "",
        ]
    else:
        lines += [
            "The following schemes survive the exact graph invariant and require the final normal-form / GL(3)^3 equivalence test:", "",
        ]
        lines += [f"- `{r['member']}` (leg permutation {r['leg_permutation']})" for r in survivors]
        lines.append("")
    lines += ["## Seven/full-pattern candidates", ""]
    for r in results:
        lines.append(f"- `{r['member']}`: edge-hist={'YES' if r['survived_edge_histogram'] else 'NO'}, graph={'YES' if r['survived_exact_graph_isomorphism'] else 'NO'}")
    (args.out / "SEED401_JKU_STRONG.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
