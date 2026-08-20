import json
from pathlib import Path

import sympy as sp

from exactify_rank23 import exact_verify


CERT = Path(__file__).parent / "results" / "blind_rank23" / "exact" / "rank23_exact.json"


def _expr_matrix(rows):
    return [[sp.sympify(x, locals={"sqrt": sp.sqrt}) for x in row] for row in rows]


def test_bundled_exact_rank23_certificate():
    obj = json.loads(CERT.read_text())
    U = _expr_matrix(obj["U"])
    V = _expr_matrix(obj["V"])
    W = _expr_matrix(obj["W"])
    c = [sp.sympify(x, locals={"sqrt": sp.sqrt}) for x in obj["c"]]
    ok, nonzero, failures = exact_verify(U, V, W, c)
    assert ok
    assert nonzero == 0
    assert failures == []


def test_exact_rank_pattern():
    obj = json.loads(CERT.read_text())
    arrays = [_expr_matrix(obj[k]) for k in ("U", "V", "W")]
    triples = []
    for r in range(23):
        ranks = []
        for arr in arrays:
            M = sp.Matrix(3, 3, [arr[i][r] for i in range(9)])
            ranks.append(M.rank())
        triples.append(tuple(ranks))
    from collections import Counter
    assert Counter(triples) == {
        (1, 1, 1): 8,
        (1, 1, 2): 6,
        (2, 1, 1): 5,
        (2, 2, 2): 4,
    }
