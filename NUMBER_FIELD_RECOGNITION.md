# Generic number-field recognition for rank-23 exactification

The original exactifier assumed that every unresolved coefficient belonged to one field
`Q(sqrt(D))`.  That was sufficient for the first discovered endpoint, but the five-seed
replication produced endpoints for which this assumption failed.

This patch keeps the geometric exactification pipeline unchanged and replaces only the
final arithmetic-recognition layer.

## Recognition strategy

1. Recognise easy rational coefficients.
2. Use high-precision PSLQ (`algdep`) to find low-degree polynomial relations for a
   deterministic sample of unresolved coordinates.
3. Try those coordinates as primitive elements.
4. If no single coordinate spans all coefficients, try small pairwise linear
   combinations.  This handles composita such as `Q(sqrt(2), sqrt(3))`, where neither
   `sqrt(2)` nor `sqrt(3)` alone generates the full field but `sqrt(2)+sqrt(3)` does.
5. Express every coefficient exactly in the power basis
   `1, alpha, ..., alpha^(d-1)` using PSLQ.
6. Verify the 729 Brent identities by exact rational polynomial arithmetic modulo the
   minimal polynomial of `alpha`.

No radical formula is required.  A degree-4 or degree-8 field is represented directly by
its minimal polynomial and the selected real embedding.

## Diagnostics

Before recognition, the expensive high-precision Newton solution is written to
`high_precision_values.json`.  Recognition writes `field_recognition_diagnostics.json`,
including sampled individual algebraic degrees, primitive-element candidates, and the
first coefficient that failed to lie in each candidate field.

## Recommended rerun

The existing incidence-gauged checkpoints can be reused.  A little extra precision is
helpful for fields of degree greater than two:

```bash
python3 batch_exactify_replication.py \
  results/replication_5seeds \
  --jku-tar schemes-exp.tgz \
  --out results/replication_5seeds/exact_endpoints \
  --dps 220 \
  --max-field-degree 8
```

There is no need for `--force` unless you want to recompute the incidence gauges too.
