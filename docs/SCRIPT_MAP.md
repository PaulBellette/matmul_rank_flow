# Script map

The repository deliberately retains the exploratory path that led to the current method.  For a public reader, however, only a small subset should be considered **paper-facing entry points**.

## Canonical discovery and verification path

| Stage | Script | Role |
|---|---|---|
| 27 -> 26 | `collision_search_3x3.py` | autonomous schoolbook collision/fusion |
| 26 -> 23 | `autonomous_state_machine_3x3.py` | frozen specialist Pareto-beam controller |
| replication | `run_5seed_replication.sh` | runs the five fixed fresh seeds and records provenance |
| endpoint analysis | `analyze_replication_endpoints.py` | polishes and classifies numerical rank-23 endpoints |
| batch exactification | `batch_exactify_replication.py` | incidence gauge + sparse-family exactification across seeds |
| single exactification | `isotropy_incidence_gauge.py`, `sparse_family_exactify.py` | numerical-to-exact pipeline for one endpoint |
| exact verification | `verify_rank23_exact.py` | standalone 729-Brent-identity verifier |
| equivalence filtering | `rank23_equivalence_search.py` | exact-invariant funnel against external corpora |
| corpus audit | `corpus_audit.py` | records what was and was not parsed before equivalence claims |
| stronger seed-401 check | `classify_401_survivors.py` | exact sandwich-incidence separation of the seven full-pattern survivors |

The independent verification implementation is under `independent_verification/`.

## Development / diagnostic scripts

These are useful research history and tests of individual mechanisms, but they are not required to understand or reproduce the frozen five-seed result:

- `analytic_ansatz.py`, `simple_algorithm_family.py`, `symbolic_simple_family.py`
- `closed_form_homotopy.py`, `symbolic_closed_form.py`
- `collision_search.py`, `curvature_flow.py`, `geometry_flow.py`, `string_flow.py`, `rankflow.py`
- `guided_cascade_3x3.py`
- `exactify_rank23.py`, `rank23_orbit_analysis.py`, `diagnose_field_failures.py`
- `rank23_reference.py`

The root `RESULTS_*.md` and strategy/patch notes document the development sequence and are best read as a lab notebook rather than as the current API.

## External-corpus utilities

- `fetch_rank23_corpora.py`
- `jku_mirror.py`, `jku_apache_mirror.py`, `jku_apache_mirror_v2.py`
- `scan_jku_tar.py`

These are deliberately separate from the scientific search so third-party corpora can remain outside the source tree.

## Suggested later cleanup

Do not do this before the paper/reproduction path is stable.  Afterwards, the flat scripts could be reorganized without changing semantics into something like:

```text
src/matmul_rank_flow/
  tensor.py
  geometry.py
  collision.py
  basin_search.py
  exactify.py
  equivalence.py
cli/
  discover.py
  exactify.py
  classify.py
experiments/
  historical development scripts
```

The current proliferation is mostly **historical layering**, not 47 independent concepts.  A public research repository can tolerate that if the canonical path is explicit.
