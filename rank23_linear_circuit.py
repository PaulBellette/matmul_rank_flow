"""Exact support and greedy linear common-subexpression analysis for rank-23 schemes.

This is deliberately an *addition-count* analysis. Scalar multiplications by exact
constants are reported separately and are not charged here.  CSE is a deterministic
greedy straight-line-program heuristic, not a proof of globally minimal additive
complexity.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from number_field_exact import SimpleNumberField

Elt = tuple[Fraction, ...]
Vector = tuple[Elt, ...]


def _frac(x) -> Fraction:
    return Fraction(str(x))


def load_field_certificate(path: Path):
    obj = json.loads(path.read_text())
    nf = obj.get("number_field") or {}
    minpoly = nf.get("minimal_polynomial_coefficients_ascending")
    if minpoly is None:
        # Older rational certificate fallback.
        minpoly = ["0", "1"]
    field = SimpleNumberField(tuple(_frac(x) for x in minpoly))

    def bank(name: str):
        key = f"{name}_power_basis"
        if key not in obj:
            raise ValueError(f"certificate lacks {key}; regenerate with current exactifier")
        return [[field.elt(_frac(q) for q in cell) for cell in row] for row in obj[key]]

    U, V, W = bank("U"), bank("V"), bank("W")
    c = [field.elt(_frac(q) for q in cell) for cell in obj["c_power_basis"]]
    return obj, field, U, V, W, c


def _support(coeffs: Iterable[Elt], field: SimpleNumberField) -> int:
    return sum(not field.is_zero(x) for x in coeffs)


def exact_linear_forms(path: Path):
    obj, field, U, V, W, c = load_field_certificate(path)
    rank = int(obj.get("rank", len(c)))
    if rank != 23:
        raise ValueError(f"expected rank 23, got {rank}")
    # U/V are 9 x 23; each column is one input linear form.
    left = [tuple(U[i][r] for i in range(9)) for r in range(rank)]
    right = [tuple(V[i][r] for i in range(9)) for r in range(rank)]
    # Each output entry is a linear form in the 23 bilinear products.
    output = [tuple(field.mul(c[r], W[i][r]) for r in range(rank)) for i in range(9)]
    return obj, field, left, right, output


def naive_bank_additions(forms: list[Vector], field: SimpleNumberField) -> int:
    return sum(max(_support(f, field) - 1, 0) for f in forms)


def _solve_linear(M: list[list[Fraction]], b: list[Fraction]) -> list[Fraction]:
    """Exact Gauss-Jordan solve for a square nonsingular rational system."""
    n = len(b)
    A = [list(M[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if A[r][col] != 0), None)
        if pivot is None:
            raise ZeroDivisionError("singular field multiplication matrix")
        A[col], A[pivot] = A[pivot], A[col]
        p = A[col][col]
        A[col] = [x / p for x in A[col]]
        for r in range(n):
            if r == col:
                continue
            q = A[r][col]
            if q:
                A[r] = [x - q * y for x, y in zip(A[r], A[col])]
    return [A[i][-1] for i in range(n)]


def field_inv(field: SimpleNumberField, a: Elt) -> Elt:
    if field.is_zero(a):
        raise ZeroDivisionError("zero has no inverse")
    d = field.degree
    cols = []
    for j in range(d):
        e = [Fraction(0)] * d
        e[j] = Fraction(1)
        cols.append(field.mul(a, field.elt(e)))
    M = [[cols[j][i] for j in range(d)] for i in range(d)]
    x = _solve_linear(M, list(field.one()))
    return field.elt(x)


def field_div(field: SimpleNumberField, a: Elt, b: Elt) -> Elt:
    return field.mul(a, field_inv(field, b))


def _vadd(field: SimpleNumberField, a: Vector, b: Vector) -> Vector:
    return tuple(field.add(x, y) for x, y in zip(a, b))


def _vscale(field: SimpleNumberField, s: Elt, v: Vector) -> Vector:
    return tuple(field.mul(s, x) for x in v)


def canonical_projective(field: SimpleNumberField, v: Vector) -> tuple[Vector, Elt]:
    """Return (canonical vector, scale) with v = scale * canonical."""
    for x in v:
        if not field.is_zero(x):
            inv = field_inv(field, x)
            return _vscale(field, inv, v), x
    raise ValueError("cannot projectivise zero vector")


@dataclass
class Term:
    coeff: Elt
    atom: int


@dataclass
class CSEBankResult:
    naive_additions: int
    greedy_cse_additions: int
    created_subexpressions: int
    final_form_additions: int
    cse_steps: list[dict]


def greedy_linear_cse(forms: list[Vector], field: SimpleNumberField) -> CSEBankResult:
    """Greedy exact linear CSE, sharing projectively identical two-term sums.

    At each step, any pair of current terms defines an exact linear subexpression.
    Projectively identical subexpressions across forms are shared.  A newly created
    subexpression costs one addition; replacing a pair by it saves one addition in
    every occurrence.  Existing atoms can also be reused at zero creation cost.
    """
    if not forms:
        return CSEBankResult(0, 0, 0, 0, [])
    nvars = len(forms[0])
    zero = field.zero()
    one = field.one()
    atoms: list[Vector] = []
    atom_by_vec: dict[Vector, int] = {}
    for i in range(nvars):
        v = tuple(one if j == i else zero for j in range(nvars))
        atom_by_vec[v] = len(atoms)
        atoms.append(v)

    def simplify_terms(terms: list[Term]) -> list[Term]:
        by_atom: dict[int, Elt] = {}
        for t in terms:
            by_atom[t.atom] = field.add(by_atom.get(t.atom, field.zero()), t.coeff)
        return [Term(c, atom) for atom, c in sorted(by_atom.items()) if not field.is_zero(c)]

    def terms_vector(terms: list[Term]) -> Vector:
        v = tuple(field.zero() for _ in range(nvars))
        for t in terms:
            v = _vadd(field, v, _vscale(field, t.coeff, atoms[t.atom]))
        return v

    current: list[list[Term]] = []
    targets: list[Vector] = []
    for form in forms:
        terms = simplify_terms([Term(c, i) for i, c in enumerate(form) if not field.is_zero(c)])
        current.append(terms)
        targets.append(form)

    naive = sum(max(len(t) - 1, 0) for t in current)
    created = 0
    steps: list[dict] = []

    while True:
        # key -> list[(form_index, term_index_a, term_index_b, replacement_scale)]
        candidates: dict[Vector, list[tuple[int, int, int, Elt]]] = {}
        for fi, terms in enumerate(current):
            for ia in range(len(terms)):
                for ib in range(ia + 1, len(terms)):
                    ta, tb = terms[ia], terms[ib]
                    vec = _vadd(field, _vscale(field, ta.coeff, atoms[ta.atom]),
                                _vscale(field, tb.coeff, atoms[tb.atom]))
                    key, scale = canonical_projective(field, vec)
                    candidates.setdefault(key, []).append((fi, ia, ib, scale))

        best = None
        for key, occs in candidates.items():
            # One replacement per form for this candidate avoids overlapping pairs.
            by_form = {}
            for occ in occs:
                by_form.setdefault(occ[0], occ)
            chosen = list(by_form.values())
            exists = key in atom_by_vec
            creation_cost = 0 if exists else 1
            gain = len(chosen) - creation_cost
            if gain <= 0:
                continue
            score = (gain, len(chosen), int(exists))
            if best is None or score > best[0]:
                best = (score, key, chosen, exists)
        if best is None:
            break

        score, key, occs, exists = best
        if exists:
            atom_id = atom_by_vec[key]
        else:
            atom_id = len(atoms)
            atoms.append(key)
            atom_by_vec[key] = atom_id
            created += 1

        # Replace from highest term index down within each form.
        for fi, ia, ib, scale in sorted(occs, key=lambda x: x[0]):
            terms = current[fi]
            # Candidate indices refer to the current list at discovery time. There is
            # only one occurrence per form in this step, so they are still valid.
            new_terms = [t for k, t in enumerate(terms) if k not in (ia, ib)]
            new_terms.append(Term(scale, atom_id))
            current[fi] = simplify_terms(new_terms)
            if terms_vector(current[fi]) != targets[fi]:
                raise AssertionError("greedy CSE replacement changed the exact linear form")
        steps.append({
            "net_addition_gain": score[0],
            "occurrences": len(occs),
            "reused_existing_atom": bool(exists),
            "atom_id": atom_id,
        })

    final_form = sum(max(len(t) - 1, 0) for t in current)
    return CSEBankResult(
        naive_additions=naive,
        greedy_cse_additions=created + final_form,
        created_subexpressions=created,
        final_form_additions=final_form,
        cse_steps=steps,
    )


def coefficient_complexity(forms: list[Vector], field: SimpleNumberField):
    zero, one, neg_one = field.zero(), field.one(), field.neg(field.one())
    counts = {"zero": 0, "plus_minus_one": 0, "other_rational": 0, "algebraic": 0}
    for form in forms:
        for x in form:
            if x == zero:
                counts["zero"] += 1
            elif x == one or x == neg_one:
                counts["plus_minus_one"] += 1
            elif all(q == 0 for q in x[1:]):
                counts["other_rational"] += 1
            else:
                counts["algebraic"] += 1
    return counts


def analyze_certificate(path: Path):
    obj, field, left, right, output = exact_linear_forms(path)
    banks = {}
    for name, forms in (("left", left), ("right", right), ("output", output)):
        cse = greedy_linear_cse(forms, field)
        banks[name] = {
            "naive_additions": cse.naive_additions,
            "greedy_cse_additions": cse.greedy_cse_additions,
            "created_subexpressions": cse.created_subexpressions,
            "final_form_additions": cse.final_form_additions,
            "cse_steps": cse.cse_steps,
            "coefficient_complexity": coefficient_complexity(forms, field),
        }
    return {
        "certificate": str(path),
        "rank": obj.get("rank"),
        "field": obj.get("field"),
        "exact_identity": obj.get("exact_identity"),
        "banks": banks,
        "naive_additions": sum(x["naive_additions"] for x in banks.values()),
        "greedy_cse_additions": sum(x["greedy_cse_additions"] for x in banks.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("certificate", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = analyze_certificate(args.certificate)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print("wrote", args.out)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
