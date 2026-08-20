# Promoted results

`runs/` is disposable exploratory output and is ignored by git.  Promote only
runs worth preserving into this directory.

For the first blind 3x3 rank-23 success, a useful layout is:

```text
results/
  blind_rank23/
    final.pt
    beam_frontier.csv
    beam_expansions.csv
    state_machine_summary.json        # if present
    exactify/
      rank23_refined.pt
      EXACTIFY_REPORT.md
      exactify_report.json
      candidate_rational.json
      candidate_algebraic.json
```

Example promotion from the successful specialist-beam run:

```bash
mkdir -p results/blind_rank23
cp runs/autonomous_state_machine_specialist/final.pt results/blind_rank23/
cp runs/autonomous_state_machine_specialist/beam_frontier.csv results/blind_rank23/ 2>/dev/null || true
cp runs/autonomous_state_machine_specialist/beam_expansions.csv results/blind_rank23/ 2>/dev/null || true
cp runs/autonomous_state_machine_specialist/state_machine_summary.json results/blind_rank23/ 2>/dev/null || true

git rev-parse HEAD > results/blind_rank23/git_commit.txt
python3 exactify_rank23.py results/blind_rank23/final.pt \
  --out results/blind_rank23/exactify \
  --compare-reference
```

Do not interpret a failed simple coefficient recognition or direct reference
match as novelty.  The next equivalence test would need to quotient the full
matrix-multiplication isotropy group.


The bundled first success now also contains:

```text
results/blind_rank23/
  exactify/       # independently refined numerical checkpoint + first recognition pass
  incidence/      # well-conditioned full-isotropy sparse gauge
  exact/          # local-family locks + exact quadratic certificate
```

`exact/rank23_exact.json` is symbolically verified; see `EXACT_RANK23_RESULT.md`.
