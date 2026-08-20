"""Progressive equivalence search for exact/rational rank-23 3x3 schemes.

The search deliberately uses a cheap-to-expensive funnel:
  1. exact factor-matrix ranks and the Heule--Kauers--Seidl f/g invariants;
  2. canonical full rank-pattern class (channel + tensor-leg permutations);
  3. a stronger sandwich-product incidence graph invariant;
  4. graph isomorphism to propose channel bijections;
  5. numerical projective-sandwich reconstruction as a final candidate test.

Differences at stages 1--3 are *proofs of inequivalence* under the standard
matrix-multiplication isotropy group. Equality is only a filter; the final
projective reconstruction is still a numerical candidate equivalence test.

Supported corpus formats:
  * this project's exact certificate: U,V,W are 9 x rank, c is length rank;
  * Perminov/FastMatrixMultiplication JSON: u,v,w are rank x 9, n=[3,3,3], m=23;
  * generic JSON with U,V,W (or u,v,w) in either 9 x 23 or 23 x 9 shape;
  * Kauers/JKU ``.exp`` files: one trilinear product per line, e.g.
    ``(a11-a12)*(b22+b23)*(c31-c32)``.

Convention: internally factors are the cyclic trace-tensor matrices (A,B,C),
so this project's output factor W is transposed to obtain C, while Perminov's
w already encodes C^T/output-dual and is interpreted as cyclic C according to
that repository's documented convention.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from functools import lru_cache
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sympy as sp

try:
    import networkx as nx
except Exception:  # pragma: no cover - CLI gives a useful error later
    nx = None


# ---------- scheme loading ----------

@dataclass
class Scheme:
    name: str
    factors: tuple[list[sp.Matrix], list[sp.Matrix], list[sp.Matrix]]
    source: str

    @property
    def rank(self) -> int:
        return len(self.factors[0])


def _expr(x: Any) -> sp.Expr:
    if isinstance(x, sp.Expr):
        return x
    if isinstance(x, (int, np.integer)):
        return sp.Integer(int(x))
    if isinstance(x, float):
        return sp.Rational(str(x))
    return sp.sympify(x)


def _matrix_columns(arr: Any, rank: int) -> list[sp.Matrix]:
    """Accept 9xr or rx9 and return r matrices reshaped row-major."""
    if len(arr) == 9 and len(arr[0]) == rank:
        cols = [[_expr(arr[i][r]) for i in range(9)] for r in range(rank)]
    elif len(arr) == rank and len(arr[0]) == 9:
        cols = [[_expr(arr[r][i]) for i in range(9)] for r in range(rank)]
    else:
        raise ValueError(f"expected 9x{rank} or {rank}x9 factor array")
    return [sp.Matrix(3, 3, c) for c in cols]


def load_scheme_json(path: Path) -> Scheme | None:
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return None

    # Ignore explicitly failed recognition candidates / non-certificates.
    if obj.get("exact_identity") is False:
        return None

    # Perminov-style full/reduced scheme.
    if all(k in obj for k in ("u", "v", "w")):
        n = obj.get("n")
        rank = int(obj.get("m", len(obj["u"])))
        if n is not None and list(n) != [3, 3, 3]:
            return None
        if rank != 23:
            return None
        A = _matrix_columns(obj["u"], rank)
        B = _matrix_columns(obj["v"], rank)
        # Repository docs state w encodes C^T/output-dual in row-major form;
        # that is the cyclic third matrix used in Tr(ABC).
        C = _matrix_columns(obj["w"], rank)
        return Scheme(path.stem, (A, B, C), str(path))

    # This project's exact certificate / generic uppercase factors.
    if all(k in obj for k in ("U", "V", "W")):
        rank = int(obj.get("rank", len(obj.get("c", [])) or 23))
        if rank != 23:
            return None
        A = _matrix_columns(obj["U"], rank)
        B = _matrix_columns(obj["V"], rank)
        Wout = _matrix_columns(obj["W"], rank)
        C = [M.T for M in Wout]  # output coefficient -> cyclic third factor
        c = obj.get("c")
        if c is not None:
            # Bake channel coefficient into the first factor; this preserves
            # all projective/rank invariants while retaining summand weights.
            if len(c) != rank:
                raise ValueError("coefficient vector length mismatch")
            A = [_expr(c[r]) * A[r] for r in range(rank)]
        return Scheme(path.stem, (A, B, C), str(path))

    return None


_EXP_SYMBOLS = {f"{prefix}{i}{j}": sp.Symbol(f"{prefix}{i}{j}")
                for prefix in "abc" for i in range(1, 4) for j in range(1, 4)}


def _linear_form_matrix(expr: sp.Expr, prefix: str) -> sp.Matrix:
    """Convert an exact linear form in a11..a33 (etc.) to a 3x3 matrix."""
    syms = [_EXP_SYMBOLS[f"{prefix}{i}{j}"] for i in range(1, 4) for j in range(1, 4)]
    expr = sp.expand(expr)
    # Reject constants/nonlinear forms.  The constant is required to vanish.
    poly = sp.Poly(expr, *syms, domain="EX")
    if poly.total_degree() > 1 or poly.TC() != 0:
        raise ValueError(f"nonlinear/nonhomogeneous {prefix}-form in .exp scheme: {expr}")
    vals = [sp.expand(expr).coeff(x) for x in syms]
    if sp.expand(expr - sum(v*x for v, x in zip(vals, syms))) != 0:
        raise ValueError(f"unsupported mixed symbols in {prefix}-form: {expr}")
    return sp.Matrix(3, 3, vals)


def _split_exp_product(line: str) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Parse one Kauers/JKU .exp trilinear summand exactly.

    The files represent the trace tensor directly as a product of one A-linear,
    one B-linear, and one C-linear form.  Scalar factors are baked into A.
    """
    expr = sp.sympify(line.replace("^", "**"), locals=_EXP_SYMBOLS)
    factors = sp.factor(expr).as_ordered_factors()
    scalar = sp.Integer(1)
    by_prefix: dict[str, sp.Expr] = {"a": sp.Integer(1), "b": sp.Integer(1), "c": sp.Integer(1)}
    for fac in factors:
        names = {str(x)[0] for x in fac.free_symbols}
        if not names:
            scalar *= fac
            continue
        if len(names) != 1 or next(iter(names)) not in by_prefix:
            raise ValueError(f"mixed A/B/C factor in .exp summand: {fac}")
        prefix = next(iter(names))
        by_prefix[prefix] *= fac
    if any(by_prefix[k] == 1 for k in "abc"):
        raise ValueError(".exp summand is missing one tensor leg")
    A = scalar * _linear_form_matrix(by_prefix["a"], "a")
    B = _linear_form_matrix(by_prefix["b"], "b")
    C = _linear_form_matrix(by_prefix["c"], "c")
    return A, B, C


def load_scheme_exp(path: Path) -> Scheme | None:
    """Load a 3x3 rank-23 Kauers/JKU ``.exp`` file.

    We intentionally accept based on content rather than filename so mirrors or
    renamed downloads work.  Files for other shapes/ranks are skipped.
    """
    try:
        lines = [x.strip() for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
    except Exception:
        return None
    if len(lines) != 23:
        return None
    # Fast shape reject before invoking SymPy: 3x3 files may only mention ij in 1..3.
    import re
    toks = re.findall(r"\b([abc])(\d)(\d)\b", "\n".join(lines))
    if not toks or any(int(i) > 3 or int(j) > 3 for _, i, j in toks):
        return None
    try:
        triples = [_split_exp_product(line) for line in lines]
    except Exception:
        return None
    A = [x[0] for x in triples]
    B = [x[1] for x in triples]
    C = [x[2] for x in triples]
    return Scheme(path.stem, (A, B, C), str(path))


def scan_corpus(root: Path) -> list[Scheme]:
    out: list[Scheme] = []
    import re
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".json", ".exp"}:
            continue
        # Public FMM corpora often contain thousands of large JSON schemes whose
        # dimensions/rank are encoded in the filename. Skip obvious non-333/23
        # entries without paying JSON parse cost.
        if p.suffix.lower() == ".json":
            m = re.search(r"(?:^|[^0-9])(\d+)x(\d+)x(\d+)_m(\d+)", p.name)
            if m and tuple(map(int, m.groups())) != (3, 3, 3, 23):
                continue
        try:
            s = load_scheme_json(p) if p.suffix.lower() == ".json" else load_scheme_exp(p)
        except Exception:
            # Corpora mix full schemes, reduced formats, manifests, and metadata.
            # Unsupported entries are intentionally skipped.
            continue
        if s is not None:
            out.append(s)
    return out


# ---------- exact rank invariants ----------

def _is_exact_zero(x: sp.Expr) -> bool:
    if x == 0:
        return True
    # Corpus entries are overwhelmingly small integers/rationals, for which
    # equality above is decisive.  Expand is enough for our exact quadratic
    # certificate and avoids a full simplify in the hot loop.
    return sp.expand(x) == 0


@lru_cache(maxsize=250_000)
def _rank3_tuple(v: tuple[sp.Expr, ...]) -> int:
    """Exact rank of nine row-major entries, cached across corpus schemes."""
    a,b,c,d,e,f,g,h,i = v
    det = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
    if not _is_exact_zero(det):
        return 3
    minors = (
        a*e-b*d, a*f-c*d, b*f-c*e,
        a*h-b*g, a*i-c*g, b*i-c*h,
        d*h-e*g, d*i-f*g, e*i-f*h,
    )
    if any(not _is_exact_zero(x) for x in minors):
        return 2
    if any(not _is_exact_zero(x) for x in v):
        return 1
    return 0


def rank3_exact(M: sp.Matrix) -> int:
    """Exact rank for a 3x3 matrix, with cross-corpus memoisation."""
    return _rank3_tuple(tuple(M))


def rank_pattern(scheme: Scheme) -> list[tuple[int, int, int]]:
    A, B, C = scheme.factors
    return [(rank3_exact(A[r]), rank3_exact(B[r]), rank3_exact(C[r])) for r in range(scheme.rank)]


def canonical_rank_pattern(pattern: Iterable[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    pattern = list(pattern)
    cands = []
    for pi in itertools.permutations(range(3)):
        cands.append(tuple(sorted((tuple(t[i] for i in pi) for t in pattern), reverse=True)))
    return min(cands)


def hks_f_terms(pattern: Iterable[tuple[int, int, int]]) -> Counter:
    out = Counter()
    for t in pattern:
        for pi in itertools.permutations(range(3)):
            out[tuple(t[i] for i in pi)] += 1
    return out


def hks_g_terms(pattern: Iterable[tuple[int, int, int]]) -> Counter:
    pat = list(pattern)
    sums = [sum(t[i] for t in pat) for i in range(3)]
    return Counter(sums)


def _fmt_monomial(exps: tuple[int, int, int]) -> str:
    p = []
    for v, e in zip("xyz", exps):
        if e == 0:
            continue
        p.append(v if e == 1 else f"{v}^{e}")
    return "".join(p) or "1"


def fmt_f(c: Counter) -> str:
    terms = []
    for e, n in sorted(c.items(), key=lambda kv: (sum(kv[0]), kv[0]), reverse=True):
        m = _fmt_monomial(e)
        terms.append(m if n == 1 else f"{n}{m}")
    return " + ".join(terms)


def fmt_g(c: Counter) -> str:
    terms = []
    for e, n in sorted(c.items(), reverse=True):
        m = f"w^{e}"
        terms.append(m if n == 1 else f"{n}{m}")
    return " + ".join(terms)


def fmt_univariate(c: Counter, var: str = "x") -> str:
    terms = []
    for e, n in sorted(c.items(), reverse=True):
        m = "1" if e == 0 else (var if e == 1 else f"{var}^{e}")
        terms.append(m if n == 1 else f"{n}{m}")
    return " + ".join(terms)


def rank_summary(scheme: Scheme) -> dict[str, Any]:
    p = rank_pattern(scheme)
    # Exact rank invariants. HKS define f(x,y,z) from symmetrised rank triples and
    # g(w) from the three leg-wise total ranks. Factor counts are a weaker
    # consequence of f; summand-rank sums are an additional cheap refinement.
    factor_counts = Counter(q for t in p for q in t)
    summand_sums = Counter(sum(t) for t in p)
    leg_sums = Counter(sum(t[i] for t in p) for i in range(3))
    # A stronger symmetric 3-variable refinement retaining each rank triple.
    sym = hks_f_terms(p)
    triple_counts = Counter(p)
    return {
        "pattern": p,
        "canonical_pattern": canonical_rank_pattern(p),
        "factor_counts": tuple(factor_counts.get(k, 0) for k in (0, 1, 2, 3)),
        "triple_counts": tuple(sorted((k, int(v)) for k, v in triple_counts.items())),
        "factor_rank_poly_terms": tuple(sorted((int(k), int(v)) for k, v in factor_counts.items())),
        "summand_rank_sum_terms": tuple(sorted((int(k), int(v)) for k, v in summand_sums.items())),
        "leg_rank_sum_terms": tuple(sorted((int(k), int(v)) for k, v in leg_sums.items())),
        "symmetric_rank_terms": tuple(sorted((k, int(v)) for k, v in sym.items())),
        "factor_rank_poly": fmt_univariate(factor_counts),
        "summand_rank_sum_poly": fmt_univariate(summand_sums),
        "leg_rank_sum_poly": fmt_univariate(leg_sums, "w"),
        "symmetric_rank_poly": fmt_f(sym),
        # Backward-compatible aliases used in the earlier exploration notes.
        "f_terms": tuple(sorted((k, int(v)) for k, v in sym.items())),
        "g_terms": tuple(sorted((int(k), int(v)) for k, v in leg_sums.items())),
        "f": fmt_f(sym),
        "g": fmt_univariate(leg_sums, "w"),
        "leg_rank_sums": tuple(sum(t[i] for t in p) for i in range(3)),
    }


# ---------- stronger exact incidence / sandwich-product graph invariant ----------

def permutation_parity(pi: tuple[int, int, int]) -> int:
    inv = sum(pi[i] > pi[j] for i in range(3) for j in range(i + 1, 3))
    return inv & 1


def leg_action(scheme: Scheme, pi: tuple[int, int, int]) -> Scheme:
    legs = scheme.factors
    newlegs = [[M.copy() for M in legs[i]] for i in pi]
    if permutation_parity(pi):
        newlegs = [[M.T for M in leg] for leg in newlegs]
    return Scheme(f"{scheme.name}@{pi}", tuple(newlegs), scheme.source)


def product_rank_labels(scheme: Scheme) -> tuple[list[tuple[int, int, int]], list[list[tuple[int, int, int]]]]:
    A, B, C = scheme.factors
    nodes = rank_pattern(scheme)
    n = scheme.rank
    edges: list[list[tuple[int, int, int]]] = [[(0, 0, 0)] * n for _ in range(n)]
    for r in range(n):
        for s in range(n):
            edges[r][s] = (
                rank3_exact(A[r] * B[s]),
                rank3_exact(B[r] * C[s]),
                rank3_exact(C[r] * A[s]),
            )
    return nodes, edges


def wl_fingerprint_for_action(scheme: Scheme, rounds: int = 8) -> str:
    nodes, edges = product_rank_labels(scheme)
    colors = [repr(x) for x in nodes]
    for _ in range(rounds):
        raw = []
        for i in range(scheme.rank):
            out = sorted((edges[i][j], colors[j]) for j in range(scheme.rank))
            inc = sorted((edges[j][i], colors[j]) for j in range(scheme.rank))
            payload = repr((colors[i], out, inc)).encode()
            raw.append(hashlib.sha256(payload).hexdigest())
        # Renormalise to deterministic compact color IDs within this graph.
        uniq = {v: k for k, v in enumerate(sorted(set(raw)))}
        new = [str(uniq[v]) for v in raw]
        if new == colors:
            break
        colors = new
    edge_hist = Counter(x for row in edges for x in row)
    payload = repr((sorted(Counter(colors).items()), sorted(edge_hist.items()))).encode()
    return hashlib.sha256(payload).hexdigest()


def canonical_wl_fingerprint(scheme: Scheme) -> tuple[str, tuple[int, int, int]]:
    vals = []
    for pi in itertools.permutations(range(3)):
        x = leg_action(scheme, pi)
        vals.append((wl_fingerprint_for_action(x), pi))
    return min(vals)


def build_graph(scheme: Scheme):
    if nx is None:
        raise RuntimeError("networkx is required for graph-isomorphism stage")
    nodes, edges = product_rank_labels(scheme)
    G = nx.DiGraph()
    for i, lab in enumerate(nodes):
        G.add_node(i, label=lab)
    for i in range(scheme.rank):
        for j in range(scheme.rank):
            G.add_edge(i, j, label=edges[i][j])
    return G


def graph_mappings(source: Scheme, target: Scheme, limit: int = 4):
    """Yield (leg permutation, source->target channel mapping) candidates."""
    if nx is None:
        return
    Gs = build_graph(source)
    nm = nx.algorithms.isomorphism.categorical_node_match("label", None)
    em = nx.algorithms.isomorphism.categorical_edge_match("label", None)
    emitted = 0
    for pi in itertools.permutations(range(3)):
        tt = leg_action(target, pi)
        Gt = build_graph(tt)
        gm = nx.algorithms.isomorphism.DiGraphMatcher(Gs, Gt, node_match=nm, edge_match=em)
        for mapping in gm.isomorphisms_iter():
            yield pi, mapping
            emitted += 1
            if emitted >= limit:
                return


# ---------- numerical projective-sandwich reconstruction ----------

def _to_float_vec(M: sp.Matrix) -> np.ndarray:
    return np.array([float(sp.N(x, 30)) for x in M], dtype=float)


def _projective_map(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, float, float] | None:
    """Fit y ~ X x in projective P^8 using a homogeneous DLT system."""
    rows = []
    for x, y in pairs:
        p = int(np.argmax(np.abs(y)))
        if abs(y[p]) < 1e-14:
            return None
        for j in range(9):
            if j == p:
                continue
            row = np.zeros(81)
            # y[p]*(Xx)_j - y[j]*(Xx)_p = 0, row-major vec(X)
            row[j * 9:(j + 1) * 9] += y[p] * x
            row[p * 9:(p + 1) * 9] -= y[j] * x
            rows.append(row)
    M = np.vstack(rows)
    _, s, vh = np.linalg.svd(M, full_matrices=False)
    X = vh[-1].reshape(9, 9)
    residual = np.linalg.norm(M @ vh[-1]) / max(np.linalg.norm(M), 1e-30)
    gap = (s[-2] / max(s[-1], 1e-300)) if len(s) >= 2 else math.inf
    return X, float(residual), float(gap)


def _kron_rearrange(X: np.ndarray) -> np.ndarray:
    """R(X) is rank one iff X = A kron B for 3x3 A,B."""
    # X indexed (a*3+b, c*3+d) = A[a,c] B[b,d]
    R = np.empty((9, 9))
    for a in range(3):
        for c in range(3):
            row = a * 3 + c
            block = X[a*3:(a+1)*3, c*3:(c+1)*3]
            R[row, :] = block.reshape(-1)
    return R


def _kron_factors(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    R = _kron_rearrange(X)
    u, s, vh = np.linalg.svd(R, full_matrices=False)
    rel = float(s[1] / s[0]) if s[0] else math.inf
    A = (u[:, 0] * math.sqrt(s[0])).reshape(3, 3)
    B = (vh[0, :] * math.sqrt(s[0])).reshape(3, 3)
    return A, B, rel


def _proportional_ratio(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return s with x ~= s*y and relative residual."""
    den = float(np.dot(y, y))
    if den < 1e-30:
        return 0.0, math.inf
    s = float(np.dot(x, y) / den)
    rel = float(np.linalg.norm(x - s * y) / max(np.linalg.norm(x), 1e-30))
    return s, rel


def projective_sandwich_test(source: Scheme, target: Scheme, mapping: dict[int, int], *, tol: float = 2e-7) -> dict[str, Any]:
    """Numerically test one fixed channel correspondence under GL(3)^3 sandwich action.

    This is a strong candidate equivalence test, not an exact proof. It fits the induced
    9-dimensional projective maps independently on all three tensor legs, checks that
    each is Kronecker-structured, then checks consistency of the recovered P,Q,R and
    all 23 weighted rank-one summands.
    """
    sf = source.factors
    tf = target.factors
    Xs = []
    fit_meta = []
    for leg in range(3):
        pairs = [(_to_float_vec(sf[leg][r]), _to_float_vec(tf[leg][mapping[r]])) for r in range(source.rank)]
        z = _projective_map(pairs)
        if z is None:
            return {"ok": False, "reason": "projective_fit_failed"}
        X, res, gap = z
        L, R, kron_rel = _kron_factors(X)
        Xs.append((X, L, R))
        fit_meta.append({"fit_residual": res, "null_gap": gap, "kron_rank2_ratio": kron_rel})

    # For A: X_A = Q^{-T} kron P -> kron factors (Q^{-T}, P) up to swap convention.
    # Our rearrangement returns first factor for outer matrix indices (a,c), i.e. Q^{-T},
    # and second factor for inner indices (b,d), i.e. P, given NumPy's row-major vec.
    # Rather than rely on labels, verify consistency through reconstructed X maps directly
    # and weighted summands below; this catches convention mistakes robustly.
    max_proj = 0.0
    product_scale_errors = []
    for r in range(source.rank):
        scales = []
        for leg in range(3):
            X = Xs[leg][0]
            x = _to_float_vec(sf[leg][r])
            y = _to_float_vec(tf[leg][mapping[r]])
            sx, rel = _proportional_ratio(X @ x, y)
            scales.append(sx)
            max_proj = max(max_proj, rel)
        product_scale_errors.append(abs(scales[0] * scales[1] * scales[2] - 1.0))

    # Each X is only determined up to a global scalar. Choose global leg scalings so their
    # product makes the median channel product scale one, then assess channel variation.
    products = []
    for r in range(source.rank):
        ss = []
        for leg in range(3):
            X = Xs[leg][0]
            x = _to_float_vec(sf[leg][r])
            y = _to_float_vec(tf[leg][mapping[r]])
            sx, _ = _proportional_ratio(X @ x, y)
            ss.append(sx)
        products.append(ss[0] * ss[1] * ss[2])
    med = float(np.median(products))
    weight_rel = max(abs(p / med - 1.0) for p in products) if med else math.inf

    max_kron = max(x["kron_rank2_ratio"] for x in fit_meta)
    max_fit = max(x["fit_residual"] for x in fit_meta)
    ok = max_proj < tol and max_kron < tol and max_fit < tol and weight_rel < 2e-5
    return {
        "ok": bool(ok),
        "projective_maps": fit_meta,
        "max_projective_residual": max_proj,
        "max_kron_rank2_ratio": max_kron,
        "weight_product_relative_spread": weight_rel,
        "global_weight_product_median": med,
    }


# ---------- progressive corpus search ----------
def _jsonable_summary(x: dict[str, Any]) -> dict[str, Any]:
    return {
        "factor_counts": list(x["factor_counts"]),
        "triple_counts": [[list(k), v] for k, v in x["triple_counts"]],
        "factor_rank_poly": x["factor_rank_poly"],
        "summand_rank_sum_poly": x["summand_rank_sum_poly"],
        "leg_rank_sum_poly": x["leg_rank_sum_poly"],
        "symmetric_rank_poly": x["symmetric_rank_poly"],
        "f": x["f"],
        "g": x["g"],
        "leg_rank_sums": list(x["leg_rank_sums"]),
        "canonical_pattern": [list(t) for t in x["canonical_pattern"]],
    }


def search(target: Scheme, corpus: list[Scheme], *, direct_limit: int = 8) -> dict[str, Any]:
    ts = rank_summary(target)
    twl, _ = canonical_wl_fingerprint(target)
    stages = Counter()
    survivors: list[dict[str, Any]] = []

    for cand in corpus:
        if Path(cand.source).resolve() == Path(target.source).resolve():
            continue
        stages["parsed_rank23"] += 1
        cs = rank_summary(cand)
        if cs["factor_counts"] != ts["factor_counts"]:
            continue
        stages["same_factor_rank_counts"] += 1
        if cs["summand_rank_sum_terms"] != ts["summand_rank_sum_terms"]:
            continue
        stages["same_summand_rank_sum"] += 1
        if cs["leg_rank_sum_terms"] != ts["leg_rank_sum_terms"]:
            continue
        stages["same_leg_rank_sum"] += 1
        if cs["symmetric_rank_terms"] != ts["symmetric_rank_terms"]:
            continue
        stages["same_symmetric_rank_poly"] += 1
        if cs["canonical_pattern"] != ts["canonical_pattern"]:
            continue
        stages["same_full_rank_pattern"] += 1
        cwl, _ = canonical_wl_fingerprint(cand)
        if cwl != twl:
            continue
        stages["same_wl_incidence"] += 1
        item = {"name": cand.name, "source": cand.source, "wl": cwl, "direct_tests": []}
        # Try graph-isomorphism channel maps and then sandwich reconstruction.
        # direct_limit=0 is useful for fast corpus triage.
        if direct_limit <= 0:
            survivors.append(item)
            continue
        for pi, mapping in graph_mappings(target, cand, limit=direct_limit):
            transformed = leg_action(cand, pi)
            test = projective_sandwich_test(target, transformed, mapping)
            item["direct_tests"].append({"leg_permutation": list(pi), "mapping": mapping, **test})
            if test.get("ok"):
                stages["numerical_equivalence_candidate"] += 1
                break
        survivors.append(item)

    return {
        "target": {"name": target.name, "source": target.source, **_jsonable_summary(ts), "wl_incidence": twl},
        "corpus_size": len(corpus),
        "stage_counts": dict(stages),
        "survivors": survivors,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    t = result["target"]
    c = result["stage_counts"]
    order = [
        "parsed_rank23", "same_factor_rank_counts", "same_summand_rank_sum",
        "same_leg_rank_sum", "same_symmetric_rank_poly", "same_full_rank_pattern",
        "same_wl_incidence", "numerical_equivalence_candidate",
    ]
    lines = [
        "# Rank-23 equivalence search",
        "",
        f"Target: `{t['source']}`",
        "",
        f"- factor-rank count polynomial (weak): `{t['factor_rank_poly']}`",
        f"- extra summand-rank-sum polynomial: `{t['summand_rank_sum_poly']}`",
        f"- HKS `g(w)`: `{t['leg_rank_sum_poly']}`",
        f"- HKS `f(x,y,z)`: `{t['symmetric_rank_poly']}`",
        f"- factor-rank counts (rank 0..3): `{t['factor_counts']}`",
        f"- sandwich-incidence WL hash: `{t['wl_incidence']}`",
        "",
        "## Funnel",
        "",
    ]
    for k in order:
        lines.append(f"- {k}: **{c.get(k, 0)}**")
    lines += ["", "## Strong-filter survivors", ""]
    if not result["survivors"]:
        lines.append("None.")
    else:
        for x in result["survivors"]:
            ok = any(z.get("ok") for z in x["direct_tests"])
            lines += [f"- `{x['source']}` — numerical equivalence candidate: **{ok}**"]
    lines += [
        "",
        "A mismatch at any exact rank/full-pattern/incidence stage is a rigorous",
        "inequivalence certificate for the compared exact schemes. A final numerical",
        "equivalence candidate should still be exactified before publication.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("certificate", type=Path)
    ap.add_argument("--corpus", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, default=Path("results/blind_rank23/equivalence_search"))
    ap.add_argument("--direct-limit", type=int, default=8)
    args = ap.parse_args()

    target = load_scheme_json(args.certificate)
    if target is None:
        raise SystemExit("could not parse target certificate")
    corpus = []
    for root in args.corpus:
        corpus.extend(scan_corpus(root))

    result = search(target, corpus, direct_limit=args.direct_limit)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "equivalence_search.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(args.out / "EQUIVALENCE_SEARCH.md", result)

    print("factor-rank polynomial:", result["target"]["factor_rank_poly"])
    print("summand-rank-sum polynomial:", result["target"]["summand_rank_sum_poly"])
    print("HKS g(w):", result["target"]["leg_rank_sum_poly"])
    print("HKS f(x,y,z):", result["target"]["symmetric_rank_poly"])
    print("target WL:", result["target"]["wl_incidence"])
    print("corpus rank-23 schemes:", len(corpus))
    for k, v in result["stage_counts"].items():
        print(f"{k}: {v}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
