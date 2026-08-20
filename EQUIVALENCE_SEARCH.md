# Rank-23 equivalence search

This stage asks a classification question, separate from the search-controller result:

> Is the exact rank-23 certificate equivalent, under the standard matrix-multiplication
> isotropy group, to a scheme already present in a known corpus?

`rank23_equivalence_search.py` uses a cheap-to-expensive funnel. All early stages use exact
arithmetic, so a mismatch is a rigorous inequivalence certificate for the two exact schemes.

1. **factor-rank counts** — the multiset of ranks of all 69 factor matrices;
2. **summand-rank sums** — a cheap extra refinement of the 23 rank triples;
3. **Heule--Kauers--Seidl `g(w)`** — the three total factor-ranks, symmetrised over tensor legs;
4. **Heule--Kauers--Seidl `f(x,y,z)`** — the symmetrised 23 factor-rank triples;
5. **full canonical rank pattern** — all 23 rank triples, modulo channel permutations and all six tensor-leg permutations;
6. **sandwich-product incidence graph** — node labels are factor-rank triples and every ordered pair of channels is labelled by
   `rank(A_r B_s), rank(B_r C_s), rank(C_r A_s)`; a canonical WL hash is used as an exact filter;
7. **exact graph isomorphism** proposes possible channel correspondences for the rare survivors;
8. **numerical projective-sandwich reconstruction** tests whether a survivor appears to be related by `GL(3)^3`.

A positive result at step 8 is deliberately called a *candidate equivalence*: before making a
publication claim, the recovered basis transforms should be exactified (or the Kauers--Moosbauer
normal form/equivalence algorithm should be run exactly).

## Supported corpus formats

The scanner recursively accepts:

- this project's exact-certificate JSON (`U,V,W,c`);
- `dronperminov/FastMatrixMultiplication` full JSON (`u,v,w,n,m`);
- generic JSON with `U,V,W` or `u,v,w` in `9 x 23` or `23 x 9` form;
- Kauers/JKU `.exp` files with one exact trilinear product per line.

The `.exp` parser is content-based rather than filename-based. Its convention has been checked
against a public Kauers `333-23` scheme by independently evaluating all 729 Brent coefficients.
Third-party schemes are **not** copied into this project.

## Corpus setup

Fetch/update current public GitHub catalogs outside the source tree:

```bash
python3 fetch_rank23_corpora.py --out external/rank23_corpora
```

Then run the exact funnel over all local trees:

```bash
python3 rank23_equivalence_search.py \
  results/blind_rank23/exact/rank23_exact.json \
  --corpus external/rank23_corpora/perminov \
  --corpus external/rank23_corpora/kauers \
  --corpus external/rank23_corpora/matmulcatalog \
  --out results/blind_rank23/equivalence_search/current_public
```

If you obtain a local mirror/download of the JKU 3x3 repository, just add it:

```bash
  --corpus external/rank23_corpora/jku
```

The JKU web repository currently reports **17,376 schemes** (last repository update shown by the
site: 2020-08-06). The helper does not pretend that saving its interactive landing page is a full
mirror.

## Interpretation

If `same_factor_rank_counts`, `same_summand_rank_sum`, `same_leg_rank_sum`,
`same_symmetric_rank_poly`, `same_full_rank_pattern`, or `same_wl_incidence` drops a candidate,
that candidate is rigorously inequivalent under the standard matrix-multiplication isotropy
symmetries handled here.

If the entire parsed corpus reaches **zero survivors**, we have a rigorous statement only about
*that parsed corpus*, not about every rank-23 scheme that exists. Heule--Kauers--Seidl found more
than 17,000 mutually inequivalent schemes, and rank-23 schemes also occur in continuous families;
a corpus miss is therefore not by itself a global novelty proof.

For survivors, the next exact step is the normal-form/equivalence machinery of Kauers--Moosbauer,
*A Normal Form for Matrix Multiplication Schemes* (arXiv:2206.00550), or exactification of the
basis maps proposed by the numerical final stage.

## Convention

Internally the tool uses the cyclic trace-tensor convention

`A -> P A Q^-1`, `B -> Q B R^-1`, `C -> R C P^-1`.

Odd permutations of tensor legs are accompanied by transpose, as required by the cyclic matrix
multiplication tensor. The project's stored output factor is transposed when converted to this
convention. Perminov's JSON documents `w` as the `C^T`/output-dual factor and is ingested in the
corresponding cyclic convention.

## References

- Heule, Kauers, Seidl, *New ways to multiply 3 x 3-matrices*, JSC 104 (2021), arXiv:1905.10192.
- Kauers, Moosbauer, *A Normal Form for Matrix Multiplication Schemes*, arXiv:2206.00550.
- https://www.algebra.uni-linz.ac.at/research/matrix-multiplication/
- https://github.com/mkauers/matrix-multiplication
- https://github.com/dronperminov/FastMatrixMultiplication

## Corpus coverage audit

Before interpreting funnel counts as a corpus-wide result, audit the downloaded repositories:

```bash
python3 corpus_audit.py \
  --corpus external/rank23_corpora/perminov \
  --corpus external/rank23_corpora/kauers \
  --corpus external/rank23_corpora/matmulcatalog \
  --out results/blind_rank23/corpus_audit/current_public
```

This writes:

- `CORPUS_AUDIT.md` — per-corpus coverage summary;
- `corpus_audit.json` — machine-readable counts and sample failures;
- `corpus_files.csv` — one row for every non-`.git` file, with an explicit status.

The equivalence search now writes the same audit files into its `--out` directory automatically.
Do not make a corpus-wide inequivalence claim while there are unresolved
`plausible_unsupported_extension`, `parse_error`, or `loader_rejected` entries.
`unsupported_reduced_json` is also a coverage gap if the reduced file is the only available
representation of a 3x3 rank-23 scheme.

## Mirroring the full JKU web repository

The GitHub repositories are not the same thing as the historical 17,376-scheme
JKU web corpus. Mirror the latter explicitly:

```bash
python3 fetch_rank23_corpora.py \
  --out external/rank23_corpora \
  --mirror-jku \
  --insecure-jku
```

`--insecure-jku` is intentionally narrow: TLS verification is disabled only for
`algebra.uni-linz.ac.at`, and redirects to other hosts are rejected. Omit it if
normal certificate verification works on your machine.

The crawler is resumable and writes `jku_manifest.jsonl`, a mirror summary,
canonical `schemes/*.exp`, and raw HTML/JS/archive responses for audit. It is
content-driven, so extensionless/query-string scheme endpoints are recognised
as schemes when they return 23 trilinear summands.

Then include the mirror in the normal audit/search:

```bash
python3 rank23_equivalence_search.py \
  results/blind_rank23/exact/rank23_exact.json \
  --corpus external/rank23_corpora/jku/schemes \
  --corpus external/rank23_corpora/perminov \
  --corpus external/rank23_corpora/kauers \
  --corpus external/rank23_corpora/matmulcatalog \
  --direct-limit 0 \
  --out results/blind_rank23/equivalence_search/full_public
```

If the mirror count is far below the site's advertised repository count, do
not infer mathematical absence. Inspect `jku_manifest.jsonl`; that means the
interactive site has an endpoint or POST/AJAX flow the crawler has not yet
discovered, and the manifest makes the remaining coverage gap explicit.
