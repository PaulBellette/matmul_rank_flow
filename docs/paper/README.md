# Paper draft

`main.tex` is a paper-facing draft of the current result.  It is intentionally more conservative than some of the exploratory notes in the repository.

The central claim in this draft is the **search method**:

- schoolbook rank 27 is reduced autonomously to rank 26 by collision/fusion;
- a frozen endpoint-free specialist Pareto controller reached rank 23 in all five fresh-seed trials;
- all five numerical endpoints polish to residuals of order `1e-15`;
- exact symbolic certificates currently verify for seeds 101, 211 and 401, plus the earlier endpoint-free rank-23 result;
- exact invariants separate those four certificates from the tested 17,376-scheme JKU archive;
- a fixed-rank complexity campaign produced exact sparse representatives at 124 and 109 naive additions for seeds 211 and 401 (82 and 71 after a deterministic exact greedy-CSE pass, with scalar constant multiplications uncharged);
- a matched complexity-guided discovery ablation showed no improvement in rank-23 discovery and slowed one weak-from-start trajectory.

The draft explicitly does **not** claim a new rank bound or previously unknown continuous rank-23 families.

Build with a conventional LaTeX installation:

```bash
cd docs/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Before submission, decide the final author list, affiliations, acknowledgements/tooling disclosure, target venue style, whether the complexity results belong in the main paper or an appendix, and whether to include the rank-22 campaign once it has a stable result.
