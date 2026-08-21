# matmul-rank-flow

Experimental search for exact low-rank bilinear algorithms by navigating the geometry of the exact tensor-decomposition manifold.

The current headline result is a frozen five-seed experiment for noncommutative `3 x 3` matrix multiplication.  Each run starts from the schoolbook rank-27 decomposition, autonomously reaches rank 26 by channel collision/fusion, and then uses the same endpoint-free specialist Pareto controller to search downward.  All five fresh seeds reached rank 23.

This is **not** a new multiplication-count record: rank 23 has been known since Laderman.  The research claim under investigation is the search method and its reproducibility.

## Paper draft

A current LaTeX draft is in [`docs/paper/main.tex`](docs/paper/main.tex).  The longer working methodology note remains at [`docs/OPTIMISER_METHODOLOGY_AND_RESULTS.md`](docs/OPTIMISER_METHODOLOGY_AND_RESULTS.md).

For the intentionally flat collection of research scripts, see [`docs/SCRIPT_MAP.md`](docs/SCRIPT_MAP.md).  The exploratory scripts are retained for provenance; the paper-facing path is much smaller.

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

See [`docs/RANK23_COMPLEXITY_SEARCH.md`](docs/RANK23_COMPLEXITY_SEARCH.md) for the objective and caveats.

## External equivalence corpora

Third-party schemes are not copied into the project.  Fetch or mirror them under ignored `external/` storage and run the exact-invariant funnel described in [`EQUIVALENCE_SEARCH.md`](EQUIVALENCE_SEARCH.md).

## Environment

Python `>=3.11`.  Project dependencies are recorded in `pyproject.toml` / `uv.lock`.

```bash
uv sync --locked
```

Exploratory output belongs under ignored `runs/`; only results worth preserving should be promoted into `results/`.
