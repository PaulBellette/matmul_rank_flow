# Rank-23 arithmetic-complexity search

This experiment keeps tensor rank fixed at 23 and asks a different question from the rank-reduction campaign:

> Once an exact rank-23 algorithm has been found, can the same geometric machinery move along the exact solution variety toward a sparser straight-line algorithm?

The first pass deliberately measures only **raw linear-form support**. For each rank-one product it counts the support of the two input linear forms, and for each of the nine outputs it counts how many of the 23 products participate in reconstruction. A `k`-term linear form contributes `k-1` additions/subtractions. This is therefore a useful proxy, not an optimal arithmetic-circuit count: common-subexpression elimination and the cost of nontrivial scalar coefficients are left for later passes.

## Why this is a constrained search

The optimizer does not minimize `tensor residual + lambda * sparsity`. Tensor equality remains a hard constraint. At each exact point it:

1. differentiates a smooth log-sum support proxy;
2. projects the negative gradient into the tangent space of exact matrix-multiplication algorithms with unit factor-column gauge;
3. takes a finite tangent step;
4. projects back to the exact constraint manifold with a Gauss--Newton corrector that preserves the starting structural zero pattern;
5. when a previously nonzero coefficient crosses the `1e-6` support threshold, attempts to snap it to an exact zero and re-correct with that zero permanently constrained;
6. keeps a tiny Pareto frontier over structural addition count, near-zero addition count, smooth support, and coefficient size.

This avoids deciding how much tensor error one missing addition is "worth".

The geometry parameterization unit-normalizes every factor column, so per-channel CP rescaling cannot create a fake sparsity gain. Existing exact zeros are frozen in both the tangent space and corrector, preventing a minimum-norm projection from slightly densifying a known sparse representation merely to improve the smooth proxy. The full `GL(3)^3` isotropy action can still change sparsity; for this experiment that is useful rather than a nuisance, because a sparser representative of the same isotropy class is genuinely a cheaper implementation. Coefficient blow-up is capped and logged.

## Overnight campaign

The default campaign starts from the two frozen-seed endpoints for which the exactification pipeline found rational representatives, seeds 211 and 401:

```bash
python3 run_rank23_complexity_campaign.py
```

Defaults:

- seeds: `211,401`
- smooth-support continuation: `tau = 0.10 -> 0.05 -> 0.02`
- 40 generations per scale
- beam width 4
- three children per retained exact state, with small tangent-space noise on two children
- structural support report: `1e-12` relative threshold
- near-zero / snap trigger: `1e-6` relative to the largest coefficient in each linear form

For a quick wiring check:

```bash
python3 run_rank23_complexity_campaign.py --smoke
```

Outputs are written under `runs/rank23_complexity_campaign/`. Each seed records `complexity_history.csv`, exact frontier checkpoints, `best_additions.pt`, `best_smooth.pt`, and `complexity_summary.json`. The campaign root records progress after each seed so a later failure does not hide an earlier result.


## First campaign result

The first campaign succeeded on both rational starting families. The numerical search reduced support-derived addition counts in both cases, and exactification then moved to still simpler exact representatives on the same constrained families:

| seed | starting exact naive adds | numerical winner | exact sparse representative | greedy exact CSE: start -> sparse | sparse exact field |
|---:|---:|---:|---:|:---|:---|
| 211 | 143 | 127 | **124** | 94 -> **82** | rational |
| 401 | 137 | 128 | **109** | 87 -> **71** | `Q(sqrt(23))` |

The seed-401 exact representative uses a quadratic generator satisfying

```text
23*alpha^2 - 4 = 0
```

and verifies all 729 Brent identities exactly. The exact-zero guardrail also passes: every coefficient deliberately frozen to zero by the campaign remains exactly zero in the symbolic certificate. Seed 211 likewise exactifies successfully with all deliberately snapped zeros preserved.

The extra simplification during exactification is important. The numerical objective found low-support neighbourhoods, but the high-precision family move could land on a substantially sparser exact point (`128 -> 109` for seed 401). This is evidence that the local exact solution family itself contains useful arithmetic structure rather than the search merely producing thresholded near-zeros.

The greedy-CSE numbers are exact deterministic straight-line-program upper bounds, not global minima. Arbitrary scalar multiplications are not charged, so `71` should not be compared directly with a published ternary `55`-addition scheme as a total scalar-operation count.

## What counts as an interesting result

The first experiment has cleared its original success criterion: it materially reduced raw additions on two independent rank-23 families and the improvements survived exactification. The next stage is therefore no longer basic feasibility; it is to improve the cost model:

1. score coefficient complexity (`0, +/-1`, small rationals, general algebraic constants);
2. charge nontrivial scalar multiplications rather than counting additions alone;
3. strengthen common-subexpression / straight-line-program optimization;
4. test whether repeated simplify -> exactify cycles continue to move toward low-cost strata;
5. compare on equal cost conventions with published low-addition rank-23 algorithms.

## Exactification and circuit follow-up

After the campaign finishes, exactify the best low-support representatives and compare them with
the original exact seed representatives:

```bash
python3 run_rank23_complexity_followup.py
```

For each seed the follow-up:

1. reads `best_additions.pt` and its explicit `frozen_zero_indices`;
2. sets exactly those coordinates to zero and chooses a zero threshold below every other factor coefficient;
3. runs `sparse_family_exactify.py` and requires exact verification of all 729 Brent identities;
4. checks that every deliberately snapped coefficient is still an exact symbolic zero;
5. computes exact naive linear-form addition counts for the original and new certificates;
6. runs a deterministic greedy exact linear CSE heuristic and reports the resulting straight-line-program upper bound.

The CSE number is **not** claimed to be the minimum additive complexity. Constant scalar
multiplications are also not included in the addition count; coefficient arithmetic is reported
separately and remains a later optimization target.
