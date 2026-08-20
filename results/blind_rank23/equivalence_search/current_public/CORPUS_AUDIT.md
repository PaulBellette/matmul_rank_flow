# Corpus audit

Every file is assigned an explicit status; nothing disappears silently.

## `/home/paul-bellete/src/matmul_rank_flow/external/rank23_corpora/perminov`

- files_seen: **6357**
- candidate_files (`.json`/`.exp` or rank-23-looking name): **2636**
- plausible_rank23_by_name: **11**
- parsed_rank23: **4**

### Status counts

- unsupported_extension: **3721**
- wrong_shape: **2579**
- wrong_rank: **44**
- parsed_rank23: **4**
- parse_error: **4**
- plausible_unsupported_extension: **3**
- unsupported_json_schema: **2**

### Extensions

- `.m`: 2757
- `.json`: 2589
- `.mpl`: 680
- `.txt`: 256
- `.exp`: 44
- `.py`: 25
- `<none>`: 2
- `.md`: 2
- `.npz`: 2

## `/home/paul-bellete/src/matmul_rank_flow/external/rank23_corpora/kauers`

- files_seen: **1785**
- candidate_files (`.json`/`.exp` or rank-23-looking name): **910**
- plausible_rank23_by_name: **1**
- parsed_rank23: **1**

### Status counts

- wrong_rank: **909**
- unsupported_extension: **875**
- parsed_rank23: **1**

### Extensions

- `.exp`: 910
- `.m`: 874
- `<none>`: 1

## `/home/paul-bellete/src/matmul_rank_flow/external/rank23_corpora/matmulcatalog`

- files_seen: **11896**
- candidate_files (`.json`/`.exp` or rank-23-looking name): **11140**
- plausible_rank23_by_name: **11**
- parsed_rank23: **7**

### Status counts

- unsupported_json_schema: **10533**
- unsupported_extension: **756**
- wrong_shape: **594**
- parsed_rank23: **7**
- parse_error: **4**
- wrong_rank: **2**

### Extensions

- `.json`: 11140
- `.java`: 477
- `.md`: 91
- `.py`: 48
- `.pdf`: 33
- `.tex`: 29
- `.out`: 22
- `.npz`: 16
- `.yml`: 6
- `<none>`: 5
- `.txt`: 5
- `.wls`: 5
- `.mpl`: 4
- `.bz2`: 3
- `.xml`: 2
- `.sh`: 1
- `.js`: 1
- `.html`: 1
- `.css`: 1
- `.bib`: 1

## Coverage rule

Before making a corpus-wide inequivalence claim, every plausible 3x3 rank-23
entry should be either `parsed_rank23` or have an explicit defensible status
such as `wrong_rank`, `wrong_shape`, or a documented unsupported format.
`plausible_unsupported_extension`, `parse_error`, and `loader_rejected` are
coverage gaps that should be resolved first.
