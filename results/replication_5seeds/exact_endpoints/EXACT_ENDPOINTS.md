# Five-seed exact endpoint classification

- endpoints attempted: **5**
- exact symbolic certificates: **3/5**

| seed | status | exact field | family nullity | family move L2 | exact factor ranks | JKU same full pattern |
|---:|:---:|:---|---:|---:|:---|---:|
| 101 | EXACT | `Q(alpha), alpha root of 499245824*t**3 - 246779995*t**2 + 434958336*t - 163881900` | 16 | 3.944e-03 | `{'0': 0, '1': 43, '2': 25, '3': 1}` | 0 |
| 211 | EXACT | `Q(alpha), alpha root of 4024823*t + 3324240` | 18 | 3.923e-03 | `{'0': 0, '1': 45, '2': 23, '3': 1}` | 0 |
| 307 | FAILED | — | — | — | — | — |
| 401 | EXACT | `Q(alpha), alpha root of 4071168*t + 2044471` | 18 | 8.177e-03 | `{'0': 0, '1': 46, '2': 22, '3': 1}` | 7 |
| 503 | FAILED | — | — | — | — | — |

## JKU scan

- parsed: **17376**
- errors: **0**

## Interpretation

A zero JKU full-pattern match is a rigorous inequivalence certificate for an exactified endpoint,
because factor-matrix ranks and their per-channel pattern are preserved by channel permutation, CP scaling,
the GL(3)^3 matrix-multiplication isotropy action, and tensor-leg permutation.

A failed exactification is not evidence against the endpoint; it means this particular sparse-family
recogniser was insufficient and the seed should be analysed separately; field-recognition diagnostics are saved per seed.
