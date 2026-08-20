# Pareto beam search over exact algorithm basins

The local differential machinery is unchanged. The global search policy is now
non-greedy.

At fixed rank R, each exact basin stores four lower-is-better intrinsic metrics:

1. soft effective nullity of the physical constraint Jacobian;
2. cheap deletion susceptibility (short bounded relaxation after setting one channel to zero);
3. local amplitude death distance |a_r| / K_r;
4. maximum coefficient amplitude.

No target nullity, known lower-rank decomposition, or target disappearing
channel is supplied.

## Frontier policy

The controller keeps a tiny Pareto beam (default width 4). A state dominates
another only when it is no worse in every metric and strictly better in at
least one. Pruning explicitly retains metric specialists before filling any
remaining beam slots by a balanced rank-sum.

Crucially, the existing frontier remains in the candidate pool every
generation. A bad hop therefore cannot erase the best low-nullity or
best-deletion basin found earlier.

Champion checkpoints are written continuously:

- `best_genericity.pt`
- `best_delete_susceptibility.pt`
- `best_death_distance.pt`

## Expansion

Only the best few frontier basins are expanded (default 2):

1. exact amplitude continuation to the local wall;
2. several exact shell-hop alternatives;
3. cheap geometric preselection;
4. deletion-susceptibility probes only on a few shortlisted children;
5. an expensive off-manifold tunnel only when exact children fail to materially
   reduce soft nullity.

Parents are retained alongside children and the combined pool is Pareto-pruned.

## Rank drop

Periodically the basin with the best deletion susceptibility receives the full
staged deletion probe. A drop is accepted only when the resulting rank-(R-1)
parameterization corrects back to exact tensor equality with bounded
coefficients. After a drop, the beam and the Jacobian smoothing scale are reset
for the new parameter dimension, but the same strategy repeats unchanged.

## CLI

Beam search is the default:

```bash
python3 autonomous_state_machine_3x3.py \
  --start path/to/exact_rankR.pt \
  --goal-rank 23 \
  --max-cycles 40 \
  --out runs/beam
```

The previous single-state controller remains available with:

```bash
--search-mode greedy
```

Useful beam controls:

```text
--beam-width 4
--beam-expand 2
--beam-exact-children 3
--beam-delete-probe-states 1
--beam-delete-every 2
--beam-delete-trigger 0.35
--beam-min-soft-gain 0.25
--beam-offhop-children 1
```

The intent is deliberately not to run a large population search. The beam is a
small memory of geometrically different exact basins so useful non-monotone
branches are not forgotten.
