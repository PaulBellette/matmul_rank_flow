# Exactifying a discovered rank-23 decomposition

The autonomous controller is numerical.  A residual around `1e-12` is excellent
search evidence but is not, by itself, an exact bilinear algorithm certificate.
The independent `exactify_rank23.py` tool is deliberately downstream of the
search machinery.

## Workflow

```bash
python3 exactify_rank23.py results/blind_rank23/final.pt \
  --out results/blind_rank23/exactify \
  --compare-reference
```

Outputs:

- `rank23_refined.pt`: independently tensor-polished numerical checkpoint;
- `EXACTIFY_REPORT.md` / `exactify_report.json`: residuals and recognition stats;
- `candidate_rational.json`: channel-pivot-gauge rational candidate;
- `candidate_algebraic.json`: small-radical recognition candidate.

If every scalar is recognised, the candidate is substituted into all 729
matrix-multiplication tensor identities and checked symbolically with SymPy.
`exact_identity: true` is the strong certificate.

## Gauge used for recognition

Each channel is first written as

`c_r * u_r ⊗ v_r ⊗ w_r`.

The largest-magnitude entry of each of `u_r`, `v_r`, `w_r` is made exactly `+1`
by signed rescaling, with the product of those three scales absorbed into
`c_r`.  This removes the trivial CP channel-scaling freedom before asking
whether coefficients look rational/simple.

This is intentionally a *small* gauge quotient.  A numerically discovered
rank-23 scheme can still be a global GL(3)-transformed version of a known scheme
and therefore look algebraically ugly in this gauge.

## Reference comparison

`--compare-reference` solves only the channel-permutation matching problem after
the same channel gauge is fixed.  It does not search the full isotropy orbit of
3x3 matrix multiplication.  Treat:

- tiny direct cost: strong evidence it is the published reference in trivial
  disguise;
- large direct cost: inconclusive; next step is a global-isotropy equivalence
  solve, not a novelty claim.

## Validation on existing checkpoints

The tool was sanity-checked on two existing rank-23 checkpoints:

1. **published exact ternary reference**
   - input/refined residual: `2.95e-15`;
   - rational recognition: `644 / 644` scalars;
   - exact symbolic verification: `true`;
   - direct reference matching cost: exactly zero.

2. **older endpoint-guided numerical rank-23 flow checkpoint**
   - input/refined residual: `4.63e-15`;
   - rational recognition: only `69 / 644` scalars;
   - small-radical recognition: `156 / 644` scalars;
   - direct reference match: mean cost about `0.745`, max about `2.12`.

The second result is an important control: a perfectly exact numerical rank-23
decomposition associated with a known-endpoint experiment can still look ugly
under the small channel gauge.  Therefore failure of simple recognition on the
blind decomposition is a reason to investigate the global isotropy orbit, not
a novelty claim.

## Full-isotropy sparse-family exactification

For the first blind rank-23 result, the small channel gauge above was only a
screening step.  The successful exactification uses two additional tools.

### 1. Incidence-derived isotropy gauge

```bash
python3 isotropy_incidence_gauge.py \
  results/blind_rank23/exactify/rank23_refined.pt \
  --out results/blind_rank23/incidence_repro
```

The script finds repeated projective directions among numerically rank-one
factor matrices and chooses well-conditioned bases for the three matrix spaces.
It applies the full continuous matrix-multiplication isotropy action (with the
third leg written in this repository's output-coefficient convention).  For the
blind result this exposes 405 structural zeros out of 644 gauge-fixed scalars.

### 2. Exactify along the reduced local family

```bash
python3 sparse_family_exactify.py \
  results/blind_rank23/incidence_repro/rank23_incidence_sparse.pt \
  --out results/blind_rank23/exact_repro
```

With structural zeros and channel pivots fixed, the reduced system has 170
unknowns and Jacobian rank/nullity 149/21.  The script uses the 21 local tangent
degrees of freedom to lock nearby simple rational coordinates, then performs an
isolated high-precision solve and number-field recognition.

For the bundled blind result this produces a complete exact certificate over

`Q(sqrt(85213608769))`

with 594 rational and 50 genuinely quadratic coefficients.  All 729 Brent
identities verify exactly.  See `EXACT_RANK23_RESULT.md`.

Standalone verification:

```bash
python3 verify_rank23_exact.py results/blind_rank23/exact/rank23_exact.json
```
