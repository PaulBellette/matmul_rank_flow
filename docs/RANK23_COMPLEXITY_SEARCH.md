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

## What counts as an interesting result

The first experiment is not expected to beat the best hand/automatically engineered low-addition rank-23 schemes. It is successful if the search can materially reduce raw additions while retaining an exact rank-23 decomposition with finite coefficients. A useful next stage would then:

1. exactify the low-support representative;
2. run common-subexpression elimination / straight-line-program optimization;
3. score coefficient complexity (`0, +/-1`, small rationals, general algebraic constants);
4. compare against published low-addition rank-23 algorithms.
