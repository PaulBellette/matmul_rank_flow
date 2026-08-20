# Rank-23 exactification report

- input: `results/blind_rank23/final.pt`
- rank: 23
- input tensor residual: `2.900872e-12`
- refined tensor residual: `4.499707e-15`
- max |amplitude| after refinement: `5.174153`

## Coefficient recognition

### rational

- recognised scalars: 69 / 644
- max scalar approximation error: `7.579e-03`
- candidate tensor residual: `5.530e-02`
- exact 729-identity verification: **False**

### algebraic

- recognised scalars: 140 / 644
- max scalar approximation error: `1.153e-06`
- candidate tensor residual: `1.716e-05`
- exact 729-identity verification: **False**

## Direct published-reference comparison

This only quotients channel permutation and per-channel CP scaling gauge; it does not search the full matrix-multiplication isotropy group.

- mean matched-channel cost: `1.254e+00`
- max matched-channel cost: `2.000e+00`

