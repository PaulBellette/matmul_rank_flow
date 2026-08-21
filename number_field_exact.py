from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import mpmath as mp
import sympy as sp


def _gcd_many(xs):
    g = 0
    for x in xs:
        g = math.gcd(g, abs(int(x)))
    return g or 1


def normalize_int_relation(rel):
    rel = [int(x) for x in rel]
    g = _gcd_many(rel)
    rel = [x // g for x in rel]
    # positive leading coefficient (highest degree / last entry)
    if rel[-1] < 0:
        rel = [-x for x in rel]
    return rel


def poly_eval_mp(coeffs, x):
    out = mp.mpf('0')
    for c in reversed(coeffs):
        out = out * x + mp.mpf(int(c))
    return out


def algdep(x, *, max_degree=10, tol=None, maxcoeff=10**18, maxsteps=200000):
    """Return a small irreducible integer polynomial for x, coeffs ascending.

    Rational/zero values are expected to be handled by the caller. PSLQ rejects
    vectors containing zero entries, so a primitive-element candidate produced by
    exact cancellation is simply not an algebraic-dependence candidate here.
    """
    if tol is None:
        tol = mp.mpf(10) ** (-(mp.mp.dps - 30))
    try:
        if not mp.isfinite(x) or abs(x) <= tol:
            return None
    except (TypeError, ValueError):
        return None
    for deg in range(2, max_degree + 1):
        vec = [mp.mpf(1)]
        for _ in range(deg):
            vec.append(vec[-1] * x)
        rel = mp.pslq(mp.matrix(vec), tol=tol, maxcoeff=maxcoeff, maxsteps=maxsteps)
        if not rel or rel[-1] == 0:
            continue
        rel = normalize_int_relation(rel)
        if abs(poly_eval_mp(rel, x)) > tol * 100:
            continue
        p = sp.Poly(sum(sp.Integer(c) * sp.Symbol('t')**i for i, c in enumerate(rel)), sp.Symbol('t'), domain=sp.QQ)
        # PSLQ can return a reducible polynomial. Choose the irreducible factor that vanishes at this embedding.
        best = None
        for factor, _mult in sp.factor_list(p)[1]:
            cc_desc = factor.all_coeffs()
            cc = [int(c) if c.q == 1 else Fraction(int(c.p), int(c.q)) for c in reversed(cc_desc)]
            # clear denominators
            den = 1
            for c in cc:
                if isinstance(c, Fraction):
                    den = math.lcm(den, c.denominator)
            ints = [int(c * den) if isinstance(c, Fraction) else int(c * den) for c in cc]
            ints = normalize_int_relation(ints)
            err = abs(poly_eval_mp(ints, x))
            if best is None or err < best[0]:
                best = (err, ints)
        if best is not None and best[0] <= tol * 100:
            return best[1]
    return None


def reconstruct_in_power_basis(x, alpha, minpoly, *, tol=None, maxcoeff=10**24, maxsteps=300000):
    """Represent x = sum_{i<d} q_i alpha^i using one PSLQ relation."""
    if tol is None:
        tol = mp.mpf(10) ** (-(mp.mp.dps - 30))
    d = len(minpoly) - 1
    basis = [mp.mpf(1)]
    for _ in range(1, d):
        basis.append(basis[-1] * alpha)
    rel = mp.pslq(mp.matrix(basis + [x]), tol=tol, maxcoeff=maxcoeff, maxsteps=maxsteps)
    if not rel or rel[-1] == 0:
        return None
    rel = normalize_int_relation(rel)
    c = int(rel[-1])
    qs = [Fraction(-int(a), c) for a in rel[:-1]]
    y = mp.fsum(mp.mpf(q.numerator) / q.denominator * basis[i] for i, q in enumerate(qs))
    err = abs(y - x)
    if err > tol * 100:
        return None
    return qs, err, rel


def frac_str(q: Fraction) -> str:
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def coeffs_expr(coeffs: Iterable[Fraction], symbol='alpha') -> str:
    terms = []
    for i, q in enumerate(coeffs):
        if not q:
            continue
        base = frac_str(abs(q))
        if i == 0:
            mono = base
        elif i == 1:
            mono = symbol if abs(q) == 1 else f"{base}*{symbol}"
        else:
            mono = f"{symbol}^{i}" if abs(q) == 1 else f"{base}*{symbol}^{i}"
        if not terms:
            terms.append(mono if q > 0 else f"-{mono}")
        else:
            terms.append((" + " if q > 0 else " - ") + mono)
    return ''.join(terms) if terms else '0'


@dataclass(frozen=True)
class SimpleNumberField:
    """Exact Q[alpha]/(p(alpha)) arithmetic in the power basis."""
    minpoly: tuple[Fraction, ...]  # ascending, includes leading coefficient

    def __post_init__(self):
        if len(self.minpoly) < 2 or self.minpoly[-1] == 0:
            raise ValueError('invalid minimal polynomial')

    @property
    def degree(self):
        return len(self.minpoly) - 1

    def zero(self):
        return tuple(Fraction(0) for _ in range(self.degree))

    def one(self):
        return (Fraction(1),) + tuple(Fraction(0) for _ in range(self.degree - 1))

    def from_rational(self, q):
        q = Fraction(q)
        return (q,) + tuple(Fraction(0) for _ in range(self.degree - 1))

    def elt(self, coeffs):
        c = [Fraction(x) for x in coeffs]
        if len(c) > self.degree:
            c = self.reduce(c)
        c += [Fraction(0)] * (self.degree - len(c))
        return tuple(c[:self.degree])

    def reduce(self, coeffs):
        c = [Fraction(x) for x in coeffs]
        d = self.degree
        if len(c) < d:
            c += [Fraction(0)] * (d - len(c))
        lead = self.minpoly[d]
        for k in range(len(c) - 1, d - 1, -1):
            a = c[k]
            if not a:
                continue
            fac = a / lead
            for i in range(d):
                c[k-d+i] -= fac * self.minpoly[i]
            c[k] = Fraction(0)
        c = c[:d]
        c += [Fraction(0)] * (d - len(c))
        return c

    def add(self, a, b):
        return tuple(x + y for x, y in zip(a, b))

    def neg(self, a):
        return tuple(-x for x in a)

    def sub(self, a, b):
        return tuple(x - y for x, y in zip(a, b))

    def mul(self, a, b):
        d = self.degree
        c = [Fraction(0)] * (2*d - 1)
        for i, x in enumerate(a):
            if x:
                for j, y in enumerate(b):
                    if y:
                        c[i+j] += x*y
        return tuple(self.reduce(c))

    def is_zero(self, a):
        return all(x == 0 for x in a)

    def determinant2(self, a,b,c,d):
        return self.sub(self.mul(a,d), self.mul(b,c))

    def determinant3(self, M):
        # a(ei-fh)-b(di-fg)+c(dh-eg)
        a,b,c = M[0]; d,e,f = M[1]; g,h,i = M[2]
        t1 = self.mul(a, self.sub(self.mul(e,i), self.mul(f,h)))
        t2 = self.mul(b, self.sub(self.mul(d,i), self.mul(f,g)))
        t3 = self.mul(c, self.sub(self.mul(d,h), self.mul(e,g)))
        return self.add(self.sub(t1,t2), t3)

    def rank3(self, M):
        if not self.is_zero(self.determinant3(M)):
            return 3
        inds = ((0,1),(0,2),(1,2))
        for I in inds:
            for J in inds:
                if not self.is_zero(self.determinant2(M[I[0]][J[0]], M[I[0]][J[1]], M[I[1]][J[0]], M[I[1]][J[1]])):
                    return 2
        return 0 if all(self.is_zero(x) for row in M for x in row) else 1


def verify_brent_power_basis(U, V, W, c, field: SimpleNumberField, n=3):
    """Verify all n^6 Brent identities exactly in a simple number field."""
    nonzero = 0
    failures = []
    # flattened 3x3 index is row-major. Target tensor: A_ij B_jk -> C_ik.
    for ai in range(n):
        for aj in range(n):
            ia = n*ai + aj
            for bi in range(n):
                for bj in range(n):
                    ib = n*bi + bj
                    for ci in range(n):
                        for cj in range(n):
                            ic = n*ci + cj
                            s = field.zero()
                            for r in range(len(c)):
                                term = field.mul(c[r], field.mul(U[ia][r], field.mul(V[ib][r], W[ic][r])))
                                s = field.add(s, term)
                            target = 1 if (aj == bi and ai == ci and bj == cj) else 0
                            if target:
                                s = field.sub(s, field.one())
                            if not field.is_zero(s):
                                nonzero += 1
                                if len(failures) < 10:
                                    failures.append((ia, ib, ic, [frac_str(x) for x in s]))
    return nonzero == 0, nonzero, failures


def discover_common_field(unresolved, *, max_degree=12, tol=None, maxcoeff_algdep=10**30,
                          maxcoeff_basis=10**40, pair_sample=20, individual_sample=64,
                          triple_sample=10, aggregate_sample=8):
    """Find a primitive alpha whose power basis spans all unresolved mp values.

    This is deliberately staged for performance. We first compute algdep relations
    for a modest sample of coordinates and immediately test promising individual
    coordinates as primitive elements. Only if none spans the field do we try
    generic pairwise linear combinations, which is the common fix for composita
    such as Q(sqrt(2),sqrt(3)).
    """
    if tol is None:
        tol = mp.mpf(10) ** (-(mp.mp.dps - 30))
    diagnostics = {"individual": [], "candidates": [], "search": {"max_degree": max_degree, "individual_sample": individual_sample, "pair_sample": pair_sample, "triple_sample": triple_sample, "aggregate_sample": aggregate_sample}}

    def try_candidate(alpha, p, label):
        d = len(p)-1
        row = {"label": label, "degree": d, "minpoly": p}
        reps = {}
        maxerr = mp.mpf('0')
        for idx, x in unresolved:
            rec = reconstruct_in_power_basis(x, alpha, p, tol=tol, maxcoeff=maxcoeff_basis)
            if rec is None:
                row["failed_index"] = int(idx)
                row["max_error"] = mp.nstr(maxerr, 10)
                diagnostics["candidates"].append(row)
                return None
            qs, err, _rel = rec
            # Decimal PSLQ with a huge coefficient budget can fit an algebraic
            # coordinate absurdly well even when it is *not* in Q(alpha). If an
            # independent minimal polynomial for this coordinate is available,
            # require the reconstructed field element to satisfy it exactly.
            kp = known_minpolys.get(int(idx))
            if kp is not None:
                fld = SimpleNumberField(tuple(Fraction(int(c), 1) for c in p))
                elt = fld.elt(qs)
                acc = fld.zero()
                for coeff in reversed(kp):
                    acc = fld.add(fld.mul(acc, elt), fld.from_rational(int(coeff)))
                if not fld.is_zero(acc):
                    row["failed_index"] = int(idx)
                    row["failure_reason"] = "reconstruction violates coordinate minimal polynomial"
                    row["max_error"] = mp.nstr(maxerr, 10)
                    diagnostics["candidates"].append(row)
                    return None
            reps[int(idx)] = qs
            maxerr = max(maxerr, err)
        row["failed_index"] = None
        row["max_error"] = mp.nstr(maxerr, 10)
        diagnostics["candidates"].append(row)
        diagnostics["selected"] = {"label": label, "degree": d, "minpoly": p, "max_error": mp.nstr(maxerr, 10)}
        return {"alpha": alpha, "minpoly": p, "representations": reps, "diagnostics": diagnostics}

    # Prefer coordinates with larger magnitude variety, but keep deterministic order.
    # Test each discovered individual generator immediately.  The old ordering first
    # ran algdep on the entire sample (up to 64 coordinates) before trying the first
    # candidate, which is needlessly expensive when an early coordinate already
    # generates the common field (as happens for simple quadratic families).
    sample_values = list(unresolved[:individual_sample])
    individual = []
    known_minpolys = {}
    tried_individual = set()
    for idx, x in sample_values:
        p = algdep(x, max_degree=max_degree, tol=tol, maxcoeff=maxcoeff_algdep)
        row = {"index": int(idx), "degree": (len(p)-1 if p else None), "minpoly": p}
        diagnostics["individual"].append(row)
        if p:
            known_minpolys[int(idx)] = p
            label = f"x[{idx}]"
            individual.append((len(p)-1, idx, x, p, label))
            tried_individual.add(label)
            got = try_candidate(x, p, label)
            if got is not None:
                deg_hist = {}
                for rr in diagnostics["individual"]:
                    deg_hist[str(rr["degree"])] = deg_hist.get(str(rr["degree"]), 0) + 1
                diagnostics["degree_histogram"] = deg_hist
                return got

    # A coordinate of maximal detected degree is most likely already primitive.
    individual.sort(reverse=True, key=lambda z: (z[0], -max(abs(c) for c in z[3])))
    for _d, _idx, x, p, label in individual:
        if label in tried_individual:
            continue
        got = try_candidate(x, p, label)
        if got is not None:
            deg_hist = {}
            for row in diagnostics["individual"]:
                deg_hist[str(row["degree"])] = deg_hist.get(str(row["degree"]), 0) + 1
            diagnostics["degree_histogram"] = deg_hist
            return got

    # No sampled coordinate spans everything. Try generic sums of the best few
    # algebraic coordinates; primitive element theorem says generic combinations
    # work for finite separable extensions.
    top = individual[:pair_sample]
    seen = set()
    for a_pos in range(len(top)):
        for b_pos in range(a_pos+1, len(top)):
            xa, ia = top[a_pos][2], top[a_pos][1]
            xb, ib = top[b_pos][2], top[b_pos][1]
            for k in (1, -1, 2, -2, 3, -3):
                beta = xa + k*xb
                key = mp.nstr(beta, 60)
                if key in seen:
                    continue
                seen.add(key)
                p = algdep(beta, max_degree=max_degree, tol=tol, maxcoeff=maxcoeff_algdep)
                if not p:
                    continue
                label = f"x[{ia}] {('+' if k>=0 else '-')} {abs(k)}*x[{ib}]"
                got = try_candidate(beta, p, label)
                if got is not None:
                    deg_hist = {}
                    for row in diagnostics["individual"]:
                        deg_hist[str(row["degree"])] = deg_hist.get(str(row["degree"]), 0) + 1
                    diagnostics["degree_histogram"] = deg_hist
                    return got

    # Pairwise primitive elements can still miss a compositum generated by three
    # or more independent subfields. Try a small deterministic triple search.
    tri = individual[:triple_sample]
    triple_coeffs = (
        (1, 1, 1), (1, 1, -1), (1, -1, 1),
        (1, 2, 3), (1, 2, -3), (1, -2, 3),
        (1, 3, 5),
    )
    for a_pos in range(len(tri)):
        for b_pos in range(a_pos + 1, len(tri)):
            for c_pos in range(b_pos + 1, len(tri)):
                xa, ia = tri[a_pos][2], tri[a_pos][1]
                xb, ib = tri[b_pos][2], tri[b_pos][1]
                xc, ic = tri[c_pos][2], tri[c_pos][1]
                for ca, cb, cc in triple_coeffs:
                    beta = ca*xa + cb*xb + cc*xc
                    key = mp.nstr(beta, 60)
                    if key in seen:
                        continue
                    seen.add(key)
                    p = algdep(beta, max_degree=max_degree, tol=tol, maxcoeff=maxcoeff_algdep)
                    if not p:
                        continue
                    label = f"{ca}*x[{ia}] {cb:+d}*x[{ib}] {cc:+d}*x[{ic}]"
                    got = try_candidate(beta, p, label)
                    if got is not None:
                        deg_hist = {}
                        for row in diagnostics["individual"]:
                            deg_hist[str(row["degree"])] = deg_hist.get(str(row["degree"]), 0) + 1
                        diagnostics["degree_histogram"] = deg_hist
                        return got

    # Final deterministic generic combinations. Primitive-element theorem says a
    # generic linear combination of generators is primitive for a finite separable
    # extension. Prime-ish weights avoid the accidental cancellations common in sums.
    agg = individual[:aggregate_sample]
    weight_sets = (
        (1,2,3,5,7,11,13,17),
        (1,-2,3,-5,7,-11,13,-17),
        (1,3,7,11,17,23,29,37),
    )
    for weights in weight_sets:
        for width in range(3, len(agg)+1):
            beta = mp.fsum(weights[j] * agg[j][2] for j in range(width))
            key = mp.nstr(beta, 60)
            if key in seen:
                continue
            seen.add(key)
            p = algdep(beta, max_degree=max_degree, tol=tol, maxcoeff=maxcoeff_algdep)
            if not p:
                continue
            label = " + ".join(f"{weights[j]}*x[{agg[j][1]}]" for j in range(width))
            got = try_candidate(beta, p, label)
            if got is not None:
                deg_hist = {}
                for row in diagnostics["individual"]:
                    deg_hist[str(row["degree"])] = deg_hist.get(str(row["degree"]), 0) + 1
                diagnostics["degree_histogram"] = deg_hist
                return got

    deg_hist = {}
    for row in diagnostics["individual"]:
        deg_hist[str(row["degree"])] = deg_hist.get(str(row["degree"]), 0) + 1
    diagnostics["degree_histogram"] = deg_hist
    raise RuntimeError("could not discover a common simple number field", diagnostics)

