# Five-seed end-to-end replication

- completed/visible seeds: **5**
- reached rank 23: **5/5**

| seed | final rank | rank-23? | drops | path | collision residual | beam gens | final residual | max |a| |
|---:|---:|:---:|---:|:---|---:|---:|---:|---:|
| 101 | 23 | YES | 3 | 26 -> 25 -> 24 -> 23 | 6.280e-16 | 12 | 2.380e-13 | 5.138 |
| 211 | 23 | YES | 3 | 26 -> 25 -> 24 -> 23 | 6.280e-16 | 12 | 3.608e-15 | 4.608 |
| 307 | 23 | YES | 3 | 26 -> 25 -> 24 -> 23 | 6.280e-16 | 25 | 3.107e-10 | 3.856 |
| 401 | 23 | YES | 3 | 26 -> 25 -> 24 -> 23 | 6.280e-16 | 12 | 8.306e-10 | 3.572 |
| 503 | 23 | YES | 3 | 26 -> 25 -> 24 -> 23 | 6.280e-16 | 9 | 3.229e-15 | 3.593 |

## Interpretation

This is intended as a frozen-policy replication: each seed independently chooses a symmetry-equivalent schoolbook collision and then runs the same specialist Pareto-beam controller. The only intended experimental variable is the RNG seed; `max_cycles` is a fixed stopping budget, not a tuned per-seed parameter.

