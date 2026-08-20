# Rank-23 orbit-invariant report

These are rank-based invariants under the full matrix-multiplication isotropy action
(invertible sandwich transformations, tensor-leg permutations/transposes, channel permutation, and CP scaling).
They are sufficient to prove inequivalence when they differ, but equality does not prove equivalence.

## Blind scheme

- factor-rank counts: `{'1': 46, '2': 23, '3': 0}`
- channel rank triples: `{'(1, 1, 1)': 8, '(1, 1, 2)': 6, '(2, 1, 1)': 5, '(2, 2, 2)': 4}`
- HKS f(x,y,z): `24x^2y^2z^2 + 22x^2yz + 22xy^2z + 22xyz^2 + 48xyz`
- leg rank sums: `[32, 27, 33]`
- HKS g(w): `w^33 + w^32 + w^27`
- max rank-1 sigma2/sigma1: `6.926e-16`
- min rank-2 sigma2/sigma1: `2.302e-02`
- max rank-2 sigma3/sigma1: `1.538e-16`

## Bundled published reference

- factor-rank counts: `{'1': 49, '2': 20, '3': 0}`
- channel rank triples: `{'(1, 1, 1)': 11, '(1, 2, 1)': 3, '(2, 1, 1)': 5, '(2, 2, 2)': 4}`
- HKS f(x,y,z): `24x^2y^2z^2 + 16x^2yz + 16xy^2z + 16xyz^2 + 66xyz`
- leg rank sums: `[32, 30, 27]`
- HKS g(w): `w^32 + w^30 + w^27`
- same full rank-pattern class: **False**

Because the rank-pattern classes differ, the two schemes are not equivalent under the full isotropy group.

