# Exact rank-23 representative from the blind search family

The specialist Pareto-beam controller reached a numerical rank-23 decomposition
without using a rank-23 endpoint during that run.  The downstream exactification
pipeline turns the independently refined checkpoint into a fully exact bilinear
algorithm.

## Numerical starting point

The blind checkpoint refined from tensor residual `2.900872e-12` to
`4.499707e-15` in one standalone correction step, with maximum channel amplitude
`5.174153`.

Naive per-channel-gauge recognition is *not* sufficient: only `69 / 644`
scalars were recognised as simple rationals, and a broad small-radical pass did
not produce an exact certificate.

## Orbit/rank fingerprint

Reshaping each factor vector as a `3 x 3` matrix gives the crisp rank-triple
multiset

```text
8 x (1,1,1)
6 x (1,1,2)
5 x (2,1,1)
4 x (2,2,2)
```

so the leg rank sums are `(32, 27, 33)` and the aggregate factor ranks are
`46` rank-one matrices plus `23` rank-two matrices.  This fingerprint differs
from the bundled published reference and is invariant under channel
permutation, per-channel CP scaling, and the usual matrix-multiplication basis
changes.

## Incidence gauge

The rank-one factor matrices reuse exact projective row/column directions.
`isotropy_incidence_gauge.py` chooses three small projective bases and applies
the correct matrix-multiplication isotropy action in this repository's output
convention:

```text
A -> P A Q^-1
B -> Q B R^-1
C_out -> P^-T C_out R^T
```

The selected transforms are well conditioned (`cond(P,Q,R)` approximately
`1.53, 1.25, 1.21`) and preserve the tensor residual at machine precision.
After canonical per-channel scaling, **405 / 644 coefficients are structural
zeros**.  The next coefficient is separated from zero: zeroing those 405 keeps
the residual at about `8e-15`, while zeroing one additional coefficient raises
it to about `1.35e-2`.

With the 405 zeros and 69 channel pivots fixed, only **170 scalar unknowns**
remain.

## Local family and exact representative

The reduced Brent Jacobian has rank/nullity

```text
149 / 21
```

with a clean singular-value gap.  The numerical solution is therefore on a
local positive-dimensional family after this gauge fixing.

`sparse_family_exactify.py` uses those 21 tangent degrees of freedom to lock 21
mobile coordinates to nearby simple rationals, correcting the other coordinates
after each lock.  The total move from the incidence-gauged blind point is tiny:

```text
L2 move       5.839686e-03
max coordinate move 3.533777e-03
```

After the 21 locks, the remaining 149-variable system is isolated and is
refined at high precision.  The full 729-equation residual reaches about
`2.8e-124`.

All coefficients then lie in the single quadratic field

```text
Q(sqrt(85213608769))
```

with **594 rational coefficients** and **50 genuinely quadratic coefficients**.
Substitution of the exact SymPy expressions into all 729 Brent identities gives
exactly zero in every case.

Therefore `results/blind_rank23/exact/rank23_exact.json` is a complete exact
rank-23 certificate.

Important wording: this exact certificate is an exact representative extremely
close to the numerically discovered point on the same local solution family.
The rational-lock step moves along that family; it is not merely a coordinate
or isotropy gauge transformation of the raw floating-point checkpoint.

## Reproduce

Starting from the independently refined blind checkpoint bundled here:

```bash
python3 rank23_orbit_analysis.py \
  results/blind_rank23/exactify/rank23_refined.pt \
  --out results/blind_rank23/orbit

python3 isotropy_incidence_gauge.py \
  results/blind_rank23/exactify/rank23_refined.pt \
  --out results/blind_rank23/incidence_repro

python3 sparse_family_exactify.py \
  results/blind_rank23/incidence_repro/rank23_incidence_sparse.pt \
  --out results/blind_rank23/exact_repro

python3 verify_rank23_exact.py \
  results/blind_rank23/exact_repro/rank23_exact.json
```

The final verifier should print:

```text
exact=True nonzero_identities=0
```
