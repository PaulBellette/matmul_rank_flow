# Current research status

This file is the short version of the current evidence. The longer derivation and development history lives in [`OPTIMISER_METHODOLOGY_AND_RESULTS.md`](OPTIMISER_METHODOLOGY_AND_RESULTS.md).

## 1. Rank discovery: established numerical result

For noncommutative `3 x 3` matrix multiplication, the frozen endpoint-free pipeline starts from schoolbook rank 27, obtains rank 26 by autonomous channel collision/fusion, and then runs the same specialist Pareto-beam controller with only the RNG seed changed.

Fresh seeds:

```text
101 211 307 401 503
```

All five reached rank 23 by the path

```text
26 -> 25 -> 24 -> 23
```

in 9--25 beam generations. All five endpoints polish to residuals of order `1e-15`.

This is a reproducibility result for the search method, **not** a new rank bound: rank 23 has been known since Laderman.

## 2. Exactification and structural classification

A floating-point endpoint is not treated as a certificate. The exactification pipeline searches an incidence-based `GL(3)^3` gauge, locks structural zeros and gauge pivots, moves along the remaining local exact family, refines at high precision, recognizes a common number field, and verifies all 729 Brent identities exactly.

Verifying frozen-seed certificates currently exist for:

| seed | field |
|---:|:---|
| 101 | cubic number field |
| 211 | rational |
| 401 | rational |

Together with the earlier blind exact certificate, these give four exact rank-23 algorithms. Exact invariants separate all four from the tested 17,376-scheme JKU archive. This does **not** prove that their positive-dimensional solution families are new.

Seeds 307 and 503 remain numerical rank-23 results; current higher-degree recognition attempts are not exact certificates.

## 3. Post-discovery arithmetic-complexity navigation

The first fixed-rank complexity campaign starts from the exact seed-211 and seed-401 representatives and moves on/near the exact rank-23 solution variety while preserving existing structural zeros and attempting to snap new zeros.

The search objective is raw support-derived addition count, not a complete hardware cost model. After exactification:

| seed | starting exact naive adds | numerical sparse winner | exact sparse representative | greedy exact CSE: start -> sparse | exact field of sparse representative |
|---:|---:|---:|---:|:---|:---|
| 211 | 143 | 127 | **124** | 94 -> **82** | rational |
| 401 | 137 | 128 | **109** | 87 -> **71** | `Q(sqrt(23))` |

For seed 401 the exact sparse representative can be written in a quadratic field with generator satisfying `23*alpha^2 - 4 = 0`. All deliberately frozen campaign zeros survive exactification, and all 729 Brent identities verify exactly.

The greedy-CSE figures are deterministic exact straight-line-program **upper bounds**, not proven minima. They count additions/subtractions but do not charge arbitrary scalar constant multiplications, so they are not directly comparable with a ternary `{-1,0,1}` low-addition scheme on total scalar-operation cost.

The important methodological result is that the discovered rank-23 family is navigable according to a secondary algebraic objective, and exactification can simplify the representative further.

## 4. Complexity during discovery: negative/neutral ablation

A matched ablation kept the rank-reduction operators, beam width, starting rank-26 checkpoint, and expansion budget fixed while allowing a weak arithmetic-simplicity signal to influence beam policy.

Results at the tested strength:

- baseline: rank 23 on 5/5 seeds;
- weak-from-start: rank 23 on 5/5 seeds;
- adaptive: rank 23 on 5/5 seeds;
- all three have median structural endpoint count 566;
- on seeds 101, 307, 401, and 503 the weak/adaptive policies reproduce the baseline generation count and endpoint metrics;
- seed 211 is the discriminating case: baseline reaches rank 23 in 12 beam generations, weak-from-start takes 29 with no complexity payoff, adaptive takes 12, and delayed-until-rank-24 takes 12.

The evidence therefore does **not** support complexity regularisation as a useful discovery objective at the tested strength. It is compatible with, and mildly supports, a staged strategy:

```text
rank discovery -> fixed-rank family navigation -> exactification/circuit simplification
```

## 5. Rank 22

A separate frozen rank-22 campaign has been explored from rank-23 starting points. No accepted `23 -> 22` crossing has been observed in the current campaign work. This is only negative search evidence and is **not** a lower bound.

Any future rank-22 claim must require finite coefficients, high-precision refinement, no border-rank blow-up/cancellation, and exact verification of all 729 Brent identities.

## 6. Next useful experiments

The clearest remaining controls/extensions are:

1. **Plain optimisation baseline.** Start from the same schoolbook decomposition and compare the frozen geometric/basin method against plain gradient descent or Adam, plus a simple sparsity-regularised baseline, under matched seeds and approximate compute budgets.
2. **Coefficient/circuit complexity.** Extend the fixed-rank complexity objective beyond support to coefficient alphabet, constant-multiplication cost, and stronger CSE/straight-line-program optimisation.
3. **4 x 4 scaling.** First ask whether the machinery can reproduce a known reduction at larger dimension before treating any unexplored boundary as meaningful.
4. **Rank 22.** Continue only as a high-upside exploratory campaign; absence of a crossing is not evidence of impossibility.

The paper-facing claim remains deliberately conservative: reproducible rank-23 discovery, exactification/classification of several discovered representatives, and evidence that exact rank-23 families can be navigated toward substantially simpler exact algorithms.
