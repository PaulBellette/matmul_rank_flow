# Frozen-policy 3x3 rank-23 -> rank-22 campaign

This experiment intentionally changes **no optimiser logic**.

The existing specialist Pareto beam is started from each polished rank-23 endpoint with:

- `goal_rank = 22`;
- a fresh RNG seed unrelated to the discovery seed;
- the same default controller parameters used in the frozen 5-seed replication;
- a default stopping budget of 120 beam generations.

## Why start from all five numerical basins?

All five rank-23 endpoints have independently passed operational matrix-multiplication tests. Three fresh endpoints (101, 211, 401) and the original endpoint also have exact certificates. The five numerical basins give the rank-22 search distinct launch geometry even where arithmetic exactification remains awkward.

The preferred start files are the common-residual polished endpoints:

`results/replication_5seeds/endpoint_analysis/polished/seed_<seed>_rank23_refined.pt`

The runner falls back to the original controller outputs if those are unavailable.

## Run

Put `run_rank22_campaign.py` in the repository root beside `autonomous_state_machine_3x3.py` and run:

```bash
python3 run_rank22_campaign.py
```

For an integration smoke test:

```bash
python3 run_rank22_campaign.py --smoke --seeds 101
```

To begin with only exactified fresh launch classes:

```bash
python3 run_rank22_campaign.py --seeds 101 211 401
```

Output goes to `results/rank22_campaign/` and is resumable at the level of separate launch classes (re-run with the desired `--seeds`).

## Scientific acceptance rule

The controller's existing deletion threshold may produce a numerical rank-22 landing at residual ~1e-9. **That is not a rank-22 algorithm claim.**

If rank 22 is reached, the runner:

1. performs a stricter same-rank numerical polish to `1e-13`;
2. saves `RANK22_CANDIDATE_POLISHED.pt`;
3. runs the independent operational verifier (no optimiser imports);
4. writes a large warning in `RANK22_CANDIDATE.json`.

A serious rank-22 claim additionally requires:

1. high-precision refinement with finite coefficients;
2. arithmetic recognition / exactification;
3. exact verification of all 729 matrix-unit output identities;
4. an independent CAS verification (SageMath preferred);
5. noncommutative operational verification.

Approximate low residual, border-rank blow-up, or coefficient divergence does not count.

## If no rank drop occurs

The campaign is still informative. `SUMMARY.md` records the best deletion susceptibility, death distance and soft effective nullity encountered from each launch basin. Comparing these trajectories tells us whether rank 23 behaves like the previous ranks but with a more distant boundary, or whether the geometry qualitatively changes at the known frontier.
