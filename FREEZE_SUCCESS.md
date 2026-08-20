# Freeze checklist: first endpoint-free 3x3 rank-23 run

The specialist-scheduled Pareto beam reached rank 23 without being given a
rank-23 endpoint.  Before further tuning, preserve the successful run and make
its numerical certificate independently inspectable.

## Promote the successful artifacts

```bash
mkdir -p results/blind_rank23
cp runs/autonomous_state_machine_specialist/final.pt results/blind_rank23/
cp runs/autonomous_state_machine_specialist/beam_frontier.csv results/blind_rank23/ 2>/dev/null || true
cp runs/autonomous_state_machine_specialist/beam_expansions.csv results/blind_rank23/ 2>/dev/null || true
cp runs/autonomous_state_machine_specialist/state_machine_summary.json results/blind_rank23/ 2>/dev/null || true

git rev-parse HEAD > results/blind_rank23/git_commit.txt
```

If the controller emitted rank-drop checkpoints or a lineage table with
slightly different filenames, promote those too.  The important rule is that
`runs/` remains scratch while `results/` contains the reproducible evidence.

## Exactify independently

```bash
python3 exactify_rank23.py results/blind_rank23/final.pt \
  --out results/blind_rank23/exactify \
  --compare-reference
```

The search result around `1e-12` residual is compelling numerical evidence, but
an exact bilinear algorithm requires an exact certificate.  The exactifier
first runs an independent tensor-only Gauss-Newton polish, then fixes the
per-channel CP scaling gauge and attempts coefficient recognition.  Recognised
candidates are substituted into all 729 Brent identities and checked with
SymPy exactly.

## Interpret the comparison conservatively

The direct published-reference comparison quotients only:

- rank-one channel permutation;
- trivial per-channel scalar gauge.

It does **not** quotient the full `GL(3)` isotropy group of matrix
multiplication.  A large direct matching cost is therefore inconclusive.
