# Independent verification bundle

Purpose: attempt to falsify the rank-23 results without importing any optimiser,
Brent-residual, exactification, or equivalence code from `matmul_rank_flow`.

## 1. Numerical operational check

This loads only `U,V,W,a` from a `.pt` checkpoint and literally executes the
23-product algorithm on random matrices and on all 81 pairs of matrix units.
It also tests inputs whose entries are noncommuting 2x2 real matrices.

```bash
python3 verify_numerical_operational.py path/to/rank23_refined.pt
```

The important feature is that it never constructs the matrix-multiplication tensor.
A shared tensor-indexing bug therefore cannot make this test pass.

## 2. Exact rational check (zero dependencies)

For exact certificates over Q (currently seeds 211 and 401):

```bash
python3 verify_rational_standalone.py path/to/rank23_exact.json
```

It uses only Python `Fraction` and checks:
- all 81 matrix-unit input pairs exactly;
- random exact rational matrix products;
- noncommutative M2(Q) entries;
- several finite fields.

## 3. Exact SymPy check

Works with the generic simple-number-field certificates and the legacy quadratic
certificate. SymPy supplies an entirely separate AlgebraicField implementation.

```bash
python3 verify_sympy_exact.py path/to/rank23_exact.json
```

This is useful immediately because it does not require Sage.

## 4. Exact SageMath check

This is the strongest independent arithmetic-stack check in the bundle:

```bash
sage verify_sage.py path/to/rank23_exact.json
```

It reconstructs the number field independently, evaluates the operational
bilinear program, tests noncommutative M2(K) inputs, and reduces generic
power-basis certificates modulo several good primes.

No `matmul_rank_flow` modules are imported.

## Suggested verification matrix

Run all applicable checks on:

1. the original exact certificate;
2. seed 101 (cubic field);
3. seed 211 (Q);
4. seed 401 (Q);
5. the polished numerical checkpoints for all five fresh seeds.

For seeds 211 and 401, agreement among the standard-library Fraction verifier,
SymPy, Sage, and the numerical operational verifier gives four substantially
independent paths.

## External-code comparison

Two useful independent public codebases are:

- Yinqi Sun's `sunyinqi0508/3by3r23-56a`, whose `verify.py` uses the same
  adversarial philosophy: Brent checks over Z / finite fields, random matrix
  products, and noncommutative matrix-valued entries.
- Andrew Perminov's `dronperminov/FastMatrixMultiplication`, an independent
  scheme representation/catalogue for small-format fast matrix multiplication.

Rather than copy their verification code into this bundle, keep those repositories
separate so that they remain genuinely independent code ancestries.

## Smoke-test the verifier itself

```bash
mkdir /tmp/indep-smoke && cd /tmp/indep-smoke
python3 /path/to/make_schoolbook_fixture.py
python3 /path/to/verify_rational_standalone.py schoolbook_rank27.json
python3 /path/to/verify_sympy_exact.py schoolbook_rank27.json
```

The fixture is the trivial 27-product schoolbook algorithm generated without any
project code.
