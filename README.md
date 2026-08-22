# matmul-rank-flow

Experimental search for exact low-rank bilinear algorithms by navigating the geometry of the exact tensor-decomposition manifold.

The current headline result is a frozen five-seed experiment for noncommutative `3 x 3` matrix multiplication. Each run starts from the schoolbook rank-27 decomposition, autonomously reaches rank 26 by channel collision/fusion, and then uses the same endpoint-free specialist Pareto controller to search downward. All five fresh seeds reached rank 23.

A second result is now established at fixed rank 23: the exact solution family can be navigated toward substantially lower-support representatives while preserving exact matrix multiplication. Starting from the exact seed-211 and seed-401 representatives, the first complexity campaign reduced naive linear-form addition counts from `143 -> 124` and `137 -> 109` after exactification. A deterministic exact greedy-CSE pass gives upper bounds `94 -> 82` and `87 -> 71`, respectively. These CSE counts do **not** charge arbitrary scalar multiplications and are not claimed optimal.

A matched ablation found no benefit from applying the same simplicity preference during rank discovery: baseline, weak, and adaptive policies all reached rank 23 on all five seeds with the same median structural endpoint count; weak-from-start guidance slowed seed 211 from 12 to 29 beam generations, while delayed guidance returned it to the 12-generation baseline path. The current evidence therefore supports a staged architecture: **discover rank first, simplify afterward**.

This is **not** a new multiplication-count record: rank 23 has been known since Laderman. The main research claim is the search method, its reproducibility, and the ability to navigate exact algorithm families according to secondary objectives.

## Paper draft

A current LaTeX draft is in [`docs/paper/main.tex`](docs/paper/main.tex).  The longer working methodology note remains at [`docs/OPTIMISER_METHODOLOGY_AND_RESULTS.md`](docs/OPTIMISER_METHODOLOGY_AND_RESULTS.md).

For a compact statement of what is currently established and what remains open, see [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md). For the intentionally flat collection of research scripts, see [`docs/SCRIPT_MAP.md`](docs/SCRIPT_MAP.md). The exploratory scripts are retained for provenance; the paper-facing path is much smaller.

## Frozen five-seed replication

The fixed fresh seeds are:

```text
101 211 307 401 503
```

Each replicate independently runs:

1. `collision_search_3x3.py` from schoolbook rank 27 to rank 26;
2. `autonomous_state_machine_3x3.py` from that seed's rank-26 checkpoint toward rank 23 using the frozen specialist Pareto beam.

The default stopping budget is 120 beam generations.  All other controller parameters are defaults.

Run:

```bash
bash run_5seed_replication.sh
```

or detach it:

```bash
nohup bash run_5seed_replication.sh > results/replication_5seeds.nohup.log 2>&1 &
echo $!
```

Watch progress:

```bash
tail -f results/replication_5seeds.nohup.log
```

Summary:

```bash
cat results/replication_5seeds/SUMMARY.md
```

The launcher records git provenance and continues to later seeds if one seed fails.

## Exactification and verification

Numerical rank-23 endpoints are not treated as mathematical certificates.  The downstream pipeline searches for a sparse well-conditioned isotropy gauge, moves along the local exact solution family to an arithmetic representative, recognizes a common number field, and verifies all 729 Brent identities exactly.

The first endpoint-free exact result is documented in [`EXACT_RANK23_RESULT.md`](EXACT_RANK23_RESULT.md).  Frozen-seed exactification results are under `results/replication_5seeds/exact_endpoints/`.

Standalone verification of an exact certificate:

```bash
python3 verify_rank23_exact.py results/blind_rank23/exact/rank23_exact.json
```

Independent verification code that does not depend on the search controller is under [`independent_verification/`](independent_verification/).

## Rank-23 arithmetic-complexity follow-up

A separate fixed-rank experiment searches the exact rank-23 solution variety for sparser linear forms, starting from the rational seed-211 and seed-401 representatives. Tensor equality remains a hard constraint; the search preserves existing structural zeros and attempts to snap newly tiny coefficients into exact zeros.

Quick wiring check:

```bash
python3 run_rank23_complexity_campaign.py --smoke
```

Overnight campaign:

```bash
python3 run_rank23_complexity_campaign.py
```

The first campaign produced exact post-search representatives at **124 naive / 82 greedy-CSE additions** for seed 211 (rational) and **109 naive / 71 greedy-CSE additions** for seed 401 (quadratic field `Q(sqrt(23))`). See [`docs/RANK23_COMPLEXITY_SEARCH.md`](docs/RANK23_COMPLEXITY_SEARCH.md) for the objective, exactification results, and caveats.

### Complexity during rank discovery

A matched ablation asks whether the same arithmetic-simplicity signal helps or hurts the rank-reduction path itself. The frozen controller remains the `off` baseline; guided variants change only beam retention/expansion policy, not the local search operators or compute budget.

```bash
python3 run_complexity_guidance_ablation.py --seeds 101 211 401 --variants baseline weak adaptive
```

The completed baseline/weak/adaptive five-seed comparison found no discovery benefit at the tested strength. Weak-from-start guidance changed one consequential trajectory (seed 211) and made it slower; delayed guidance recovered the baseline path. See [`docs/COMPLEXITY_GUIDED_DISCOVERY.md`](docs/COMPLEXITY_GUIDED_DISCOVERY.md).

## External equivalence corpora

Third-party schemes are not copied into the project.  Fetch or mirror them under ignored `external/` storage and run the exact-invariant funnel described in [`EQUIVALENCE_SEARCH.md`](EQUIVALENCE_SEARCH.md).

## Environment

Python `>=3.11`.  Project dependencies are recorded in `pyproject.toml` / `uv.lock`.

```bash
uv sync --locked
```

Exploratory output belongs under ignored `runs/`; only results worth preserving should be promoted into `results/`.
