# Sage verifier invocation fix

The original verifier was packaged as a `.py` file but the suggested command
used `sage verify_sage.py ...`. Sage's documented Python-script route is
`sage -python script.py`, while `.sage` files are run with `sage script.sage`.

The original file also contained one preparser-only expression (`t^2`).
This replacement uses ordinary Python syntax (`t**2`) throughout.

Recommended with the conda environment activated:

```bash
conda activate sage
python independent_verification/verify_sage.py path/to/rank23_exact.json
```

A quick environment check is:

```bash
python -c "from sage.all import QQ; print(QQ(2)/3)"
```

which should print `2/3`.
