# Zero-candidate PSLQ guard

This patch fixes a crash in the generic primitive-element search when a trial
linear combination cancels exactly to zero (for example `x_i - x_j = 0`).

`mpmath.pslq` rejects a relation vector containing zero entries. `algdep()` now
treats zero/non-finite trial primitive elements as rejected candidates and lets
the field search continue.

Regression:
- repeated algebraic coordinates force an exact-zero pair candidate;
- the search skips it and still discovers the degree-4 compositum;
- `test_number_field_exact.py`: 5/5 pass.

Replace the repo's `number_field_exact.py` with the patched file, then rerun the
307/503 diagnostics and exactification.
