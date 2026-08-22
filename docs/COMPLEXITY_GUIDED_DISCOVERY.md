# Complexity-guided rank discovery

This is a matched ablation of the frozen 3x3 rank-reduction controller: does a
weak preference for simpler intermediate linear forms help or hurt the path from
the first rank-26 collision basin down to rank 23?

The frozen result remains `complexity_mode=off`. Guided variants are experimental.

## Why the comparison starts at rank 26

The existing rank-27 -> rank-26 stage chooses one of the best schoolbook
collision pairs and embeds the same 2x2 rank-7 primitive. The eligible best
pairs are related by independent schoolbook index relabellings, so their support
scores are the same up to permutation. There is no meaningful complexity choice
before the shared rank-26 checkpoint.

The ablation therefore generates that checkpoint once per seed and gives the
identical file to every controller variant.

## What is regularised

The local search operators are **unchanged**. Guided and baseline runs generate
candidates using the same continuation, exact shell hops, off-manifold tunnels,
delete probes, beam width, and parent-expansion budget.

For every exact basin we additionally measure:

- a gauge-controlled smooth support score, using the same objective as the
  fixed-rank rank-23 complexity search;
- an "effective addition" count using a coarse relative threshold (default
  `1e-3`) so tiny temporary coefficients are not treated as fully structural.

When complexity is active, these metrics only affect beam policy:

1. they weakly bias the balanced priority ordering among retained basins; and
2. on generations where the baseline already spends its periodic fourth slot
   on exploration, the guided run spends that same slot on the simplest retained
   basin instead.

Complexity is deliberately **not** a fifth guaranteed specialist slot. The shared
rank-26 endpoint is already extremely sparse, so guaranteeing the sparsity
champion would merely preserve the starting basin and shrink the effective beam.

Above rank 23 this ablation does not freeze or snap zeros and does not trade Brent
residual against sparsity in a scalar loss. That keeps it a clean test of policy
regularisation rather than a new move operator.

## Variants

- `baseline`: frozen controller (`complexity_mode=off`).
- `weak`: constant complexity policy weight at ranks 26, 25, and 24.
- `delayed`: no complexity bias until rank <= 24.
- `adaptive`: with goal rank 23, use 1/4 of the configured weight at rank 26,
  1/2 at rank 25, and the full weight at rank 24.

Default policy weight is `0.75`; default smooth-support scale is `tau=0.08`.

## Run

Tests:

```bash
python3 -m pytest -q test_complexity_guided_discovery.py
```

Full five-seed matched campaign:

```bash
python3 run_complexity_guidance_ablation.py
```

A cheaper first pass:

```bash
python3 run_complexity_guidance_ablation.py \
  --seeds 101 211 401 \
  --variants baseline weak adaptive
```

The driver writes `progress.json`, `summary.csv`, and `SUMMARY.md` after every
completed seed/variant and reuses the shared collision checkpoint on restart.
Outputs go under `runs/complexity_guidance_ablation/` by default.

## Readout

The primary question is discovery, not final circuit optimality:

1. success rate reaching rank 23;
2. beam generations and rank-drop path;
3. runtime;
4. effective (`1e-3` by default) and structural (`1e-12`) raw addition counts
   at the first/final endpoint reached by the discovery controller.

Any rank-23 endpoint can then be passed through the separate fixed-rank
complexity cleanup and exactification pipeline. Keeping those stages separate
lets us distinguish "simplicity helped discovery" from "simplicity is easy to
optimise after discovery".

If guidance hurts, that supports a staged architecture: discover rank first,
then simplify. If weak/adaptive guidance improves success or reaches simpler
rank-23 endpoints without reducing success, it is evidence that arithmetic
simplicity carries useful information during basin navigation itself.


## Observed result

The completed baseline/weak/adaptive five-seed comparison gave:

| seed | baseline gens | weak gens | adaptive gens | delayed gens tested | structural adds at rank 23 |
|---:|---:|---:|---:|---:|---:|
| 101 | 12 | 12 | 12 | -- | 566 |
| 211 | 12 | **29** | 12 | 12 | 566 |
| 307 | 25 | 25 | 25 | 25 | 566 |
| 401 | 12 | 12 | 12 | -- | 566 |
| 503 | 9 | 9 | 9 | 9 | 566 |

All baseline, weak, and adaptive runs reached rank 23 on all five seeds. Delayed guidance was run on the three discriminating/available cases shown above and also reached rank 23. Median structural endpoint additions are 566 for baseline, weak, and adaptive; no guided policy produced a simpler first rank-23 endpoint under this metric.

Seed 211 is the only consequential divergence in this campaign. Constant weak complexity pressure increased the discovery path from 12 to 29 beam generations and produced no endpoint-complexity benefit. Turning the pressure on only at rank 24 restored the 12-generation baseline path, as did the rank-adaptive schedule.

## Interpretation

At the tested policy strength, arithmetic simplicity is **not a useful discovery objective**. In four of five seeds it is effectively neutral; in one seed, applying it too early is harmful. The delayed result localizes that harm to the earlier part of the rank-26/rank-25 search rather than to the final rank-24 -> rank-23 transition.

This contrasts sharply with the fixed-rank experiment, where explicit post-discovery complexity navigation produces large exact improvements. The current empirical architecture is therefore staged:

```text
find lower rank first
    -> navigate the resulting exact family for simplicity
    -> exactify / optimize the linear circuit
```

This is a negative/neutral ablation rather than a universal statement that simplicity can never help rank discovery. It only says that this matched weak policy did not help on the frozen five-seed evaluation and demonstrably slowed one trajectory.
