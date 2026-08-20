# Exploratory endpoint-free hybrid state machine

This stage packages the manually observed rank-reduction dynamics into a small
hybrid controller.  It intentionally does **not** use the independent rank-23
endpoint used by `guided_cascade_3x3.py`.

## Controller states

1. **CONTINUE_TO_WALL**
   - score channels by the local death-distance estimate `|a_r| / K_r`;
   - follow the exact physical constraint manifold while decreasing the selected
     amplitude;
   - stop when the amplitude reaches the empirical `|a| ~= 1` wall, local
     killability collapses, or the exact corrector stalls.

2. **HOP**
   - sample exact tangent directions;
   - score them by second-order non-integrability, the component of
     `D^2 G[q,q]` outside the range of the physical constraint Jacobian;
   - make a finite predictor step and solve for an exact algorithm on a fixed
     radius shell so the corrector cannot simply fall back to the source point;
   - among finite-coefficient landings, prefer smaller estimated death distance
     and lower tangent dimension.

3. **DELETE_PROBE**
   - temporarily leave the exact manifold;
   - clamp the selected amplitude through fractions `0.8, 0.6, 0.4, 0.2, 0`;
   - relax the remaining tensor residual with Adam while guarding coefficient
     magnitudes;
   - accept only if the zero-amplitude state enters a small-residual basin.

4. **DROP**
   - remove the zero channel;
   - correct directly on the rank-`R-1` exact tensor manifold;
   - reject the drop if exactness or the finite coefficient guard fails.

The state machine records every phase to CSV/PT checkpoints for inspection.

## Short probe

Starting from the exact autonomous rank-26 checkpoint:

- local continuation selected a live Strassen-block channel and moved
  `a: 2.0 -> 1.03` while retaining tensor residual around machine precision;
- the next blind radius-2.5 shell hop found a different exact rank-26 branch:
  physical tangent dimension fell from `123` to `84` and a new best channel had
  approximately `a=1.21`, `K=0.538`;
- continuing that channel again reached the `a ~= 1` wall;
- another shell hop produced a state with approximately `a=1.27`, `K=0.627`,
  but the deliberately short deletion probe remained at the obvious
  missing-product residual near `1.0`.

This is a useful controller smoke/probe rather than a claimed autonomous
`26 -> 25` reproduction.  It demonstrates that the state machine reproduces
both observed motions—exact wall following and finite exact basin hops—and
properly rejects an unsuccessful off-manifold delete instead of forcing a rank
drop.

Run:

```bash
python3 autonomous_state_machine_3x3.py \
  --start reference_guided_cascade/rank26.pt \
  --goal-rank 23 \
  --out runs/autonomous_state_machine
```

A tiny wiring check is available with `--smoke`.
