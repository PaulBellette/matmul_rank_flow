# Specialist-scheduled Pareto beam

## Why this patch exists

The first Pareto beam fixed *retention*: useful basins were no longer lost merely
because another basin won a balanced scalar score.  A rank-24 exploratory run
then showed a new failure mode: specialists could survive in the frontier for
many generations without being expanded.

Examples from that run included separate basins with roughly:

- lowest effective nullity,
- best cheap deletion susceptibility,
- best local death distance.

That is a valid Pareto frontier, but retention alone is not exploration.

## Generic policy

At every rank `R`, each generation explicitly identifies:

1. **genericity champion** — lowest smooth/effective Jacobian nullity;
2. **deletion champion** — lowest cheap deletion-susceptibility residual;
3. **death champion** — lowest local `|a| / K` distance-to-death estimate;
4. **explorer** — a lightly expanded/new basin when the periodic exploration
   slot is available.

If one basin owns several specialist titles, the roles are merged and the freed
slot is given to a lightly explored state.  This prevents duplicate work while
still allocating compute across qualitatively different forms of progress.

No target nullity, target channel, target decomposition, or target rank-specific
constant is used.

## Specialist operators

All specialists retain the same local differential machinery.  Only the global
compute allocation changes.

- `genericity`: local exact step + exact shell children, then proactively spend
  extra off-manifold tunnel budget to open soft Jacobian modes;
- `deletion`: local rearrangement and susceptibility probes; tunnel when local
  children fail to improve deletion susceptibility;
- `death`: continue toward the amplitude wall and use the existing hop fallback;
- `explore`: ordinary mixed expansion.

The frontier width remains four by default.  Three specialist expansions are
scheduled each generation; every third generation an additional exploration
slot may be used if beam capacity allows.

## Exact polishing

After Pareto pruning, retained states are tensor-polished toward a tighter
`1e-13` residual target before geometry is archived.  Their existing cheap
susceptibility score is preserved rather than rerun, because re-running that
noisy Adam probe would add both compute and measurement noise for a tiny
Gauss-Newton cleanup.

## Provenance

`beam_expansions.csv` records:

- generation and rank,
- basin and parent ids,
- specialist role(s),
- expansion count,
- soft nullity,
- deletion susceptibility,
- death distance,
- basin source.

Each `BasinState` also carries `expansion_count` and
`last_expanded_generation`.  Champion checkpoints save these fields too.

## CLI

The normal command is unchanged:

```bash
python3 autonomous_state_machine_3x3.py \
  --start <rank-R-checkpoint.pt> \
  --goal-rank 23 \
  --max-cycles 40 \
  --out runs/specialist_beam
```

Relevant optional controls:

```text
--beam-width 4
--beam-expand 3
--beam-explore-every 3
--beam-genericity-offhop-children 2
```

The old greedy controller remains available via `--search-mode greedy`.

## Validation

- controller/algebra/geometry group: 27 passed;
- heavier 3x3/string/guided group: 8 passed;
- compact end-to-end beam smoke: successful, with role logging and champion
  checkpointing exercised.
