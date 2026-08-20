# Post-exactification follow-up

This patch does two independent downstream jobs. It does not change the discovery controller.

## 1. Seed 401 versus its seven JKU full-pattern matches

Run:

```bash
python3 classify_401_survivors.py \
  results/replication_5seeds/exact_endpoints/seed_401/exact/rank23_exact.json \
  schemes-exp.tgz \
  --out results/replication_5seeds/exact_endpoints/seed_401/jku_strong
```

The tool first recovers the seven schemes with the same canonical full factor-rank pattern.
It then forms the exact directed coloured graph with channel node label
`(rank A_r, rank B_r, rank C_r)` and edge label

```
(rank(A_r B_s), rank(B_r C_s), rank(C_r A_s)).
```

It checks graph isomorphism for all six tensor-leg actions. A mismatch is a rigorous
inequivalence certificate; a graph-isomorphic survivor still needs the final normal-form/
GL(3)^3 test.

Result files:

- `SEED401_JKU_STRONG.md`
- `seed401_jku_strong.json`

## 2. Seeds 307 and 503 arithmetic diagnosis / retry

The generic field search is broadened from individual/pair generators to triple and generic
multi-coordinate primitive elements. It also validates reconstructed coordinates against any
independently discovered minimal polynomial *inside the candidate number field*, preventing
high-coefficient PSLQ approximations from masquerading as exact field membership.

First, the cheap diagnostic reuses the saved high-precision values and does not rerun Newton:

```bash
python3 diagnose_field_failures.py \
  results/replication_5seeds/exact_endpoints/seed_307/exact/high_precision_values.json \
  --max-degree 16

python3 diagnose_field_failures.py \
  results/replication_5seeds/exact_endpoints/seed_503/exact/high_precision_values.json \
  --max-degree 16
```

Then retry only those two endpoints:

```bash
python3 batch_exactify_replication.py \
  results/replication_5seeds \
  --jku-tar schemes-exp.tgz \
  --out results/replication_5seeds/exact_endpoints \
  --seeds 307 503 \
  --dps 220 \
  --max-field-degree 16 \
  --field-algdep-maxcoeff 100000000000000000000000000000000000000000000000000 \
  --field-basis-maxcoeff 1000000000000000000000000000000000000000000000000000000000000 \
  --field-individual-sample 128 \
  --field-pair-sample 28 \
  --field-triple-sample 12 \
  --field-aggregate-sample 10
```

The existing exact certificates for 101/211/401 are not recomputed.
