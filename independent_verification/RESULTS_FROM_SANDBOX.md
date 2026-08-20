# Independent-verifier smoke results

The verifier code was tested without importing `matmul_rank_flow`.

## Separate schoolbook fixture

A fresh 27-product schoolbook certificate was generated independently.

- standard-library Fraction operational matrix-unit check: PASS
- random exact rational products: PASS
- noncommutative M2(Q) products: PASS
- finite fields GF(2), GF(3), GF(5), GF(7), GF(11), GF(101): PASS
- SymPy exact operational check: PASS

## Existing blind rank-23 numerical endpoint

Using only U,V,W,a from `rank23_refined.pt` and ordinary NumPy matrix multiplication:

- all 81 matrix-unit input pairs: max absolute error 6.586e-13
- 500 random scalar trials: max absolute error 7.725e-12
- max relative error 2.098e-12
- 100 noncommutative M2(R) trials: max absolute error 1.097e-11
- RESULT: PASS at 1e-8 tolerance

## Existing blind rank-23 exact quadratic certificate

Using SymPy's independent `QQ<sqrt(85213608769)>` AlgebraicField implementation:

- all 81 matrix-unit pairs / 729 output coefficients: exact PASS
- 20 random exact products: PASS
- 5 noncommutative M2(K) products: PASS

This is an operational verification: the checker computes the bilinear algorithm's
output and compares it with ordinary matrix multiplication. It does not call the
project's Brent-equation verifier.
