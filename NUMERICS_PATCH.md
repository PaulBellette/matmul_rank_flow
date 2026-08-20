# Numerical robustness patch

This patch hardens the autonomous 3x3 state machine against SVD convergence
failures on highly clustered / ill-conditioned Jacobian spectra.

## Changes

1. `geometry_flow.robust_svd`
   - scales matrices to O(1) before decomposition;
   - tries `torch.linalg.svd` first;
   - on convergence failure, falls back to SciPy LAPACK `gesvd`;
   - preserves the original singular-value scale and existing `rcond` cutoff.

2. All state-machine SVD solves now use `robust_svd` through
   `geometry_flow.py`, `curvature_flow.py`, and
   `autonomous_state_machine_3x3.py`.

3. A spectrally unanalysable continuation candidate is rejected and the
   amplitude step shrunk instead of aborting the run.

4. The controller writes `latest_state.pt` before every cycle and
   `latest_exact.pt` whenever the current state is exact to numerical tolerance.

5. Added a regression test that deliberately makes PyTorch SVD throw and checks
   the conservative fallback.

## Resume the pre-patch crashed run

The old run crashed during cycle 24 while examining a candidate generated from
cycle 23.  Its last committed exact state should therefore be:

```bash
python3 autonomous_state_machine_3x3.py \
  --start runs/autonomous_state_machine_continue/cycle_23_hop.pt \
  --goal-rank 23 \
  --max-cycles 40 \
  --out runs/autonomous_state_machine_continue2
```

After this patch, future runs can normally resume from `latest_exact.pt`.
