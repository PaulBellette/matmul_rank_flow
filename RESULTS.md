# First dynamics results

These are sanity-check runs on the `2 x 2` matrix multiplication tensor.

## Schoolbook rank-8 point

```bash
uv run python geometry_flow.py --mode geometry
```

Observed:

- exact residual: `0` in the direct geometry coordinates;
- Jacobian shape: `64 x 104`;
- numerical Jacobian rank: `56`;
- nullity: `48`;
- positive singular values span only about `[1, sqrt(2)]`;
- every channel amplitude has essentially zero first-order killability.

So the schoolbook point is *not* stiff in the ordinary positive singular modes.
Instead it is a singular/high-nullity point where the exact tangent space has no
amplitude component.  That explains why directly forcing one schoolbook product
to die produces a wall.

## Sideways / curvature probe

A finite exact sideways move creates amplitude freedom.  Random tangent probes
of increasing size gave approximate best killabilities:

| sideways size | best post-correction killability |
|---:|---:|
| 0.03 | 0.018 |
| 0.10 | 0.055 |
| 0.20 | 0.113 |
| 0.40 | 0.226 |
| 0.80 | 0.383 |

This is evidence that amplitude death is curvature-dominated near the symmetric
schoolbook point.

## Exact amplitude continuation after escape

With `--escape-size 0.8 --escape-trials 60 --seed 1`, the best exact sideways
candidate had killability about `0.386`.  Exact predictor/corrector continuation
then reduced the selected normalized channel amplitude approximately

```text
1.095 -> 1.070 -> 1.045 -> 1.020 -> 1.008 -> 1.00005 -> 1.00001 -> 1.00000
```

while keeping the tensor residual around machine precision.  Killability then
collapsed and the corrector could no longer make a meaningful amplitude step.

This suggests the immediate continuation branch bends back toward an amplitude
near `1`, rather than flowing monotonically to the rank-7 boundary.  It is not a
proof of disconnected components; it is a useful geometric failure mode.

## Opposite-end sanity check

A random continuous rank-7 fit reaches essentially zero residual.  Embedding it
inside rank 8 by adding a zero-amplitude eighth channel gives very different
geometry: the seven live channel amplitudes are nontrivial (roughly magnitudes
`1.2--3.6`) and have substantial local killabilities (`~0.32--0.82`).

So the low-rank boundary is numerically healthy from its own side.  The hard
part appears to be finding a route from the symmetric schoolbook region to that
region of algorithm space, not merely resolving a badly conditioned rank-7
solution.

## Second-order curvature result

After explicitly fixing the unit-norm gauge, the schoolbook point has a
24-dimensional physical exact tangent space (`J: 88 x 104`, numerical rank 80).
For every schoolbook channel the second-order amplitude curvature operator is
PSD of rank 3, with nonzero eigenvalues numerically `(1, 1, 1)`.  There are no
negative-curvature amplitude directions.

This sharpens the earlier first-order observation: `a_r=1` is not only
stationary for every schoolbook multiplication channel, it is a second-order
local floor along the exact-algorithm manifold.

A deterministic curvature eigenvector is substantially better than random
sideways probing at creating future first-order motion.  For channel 0 a
second-order predictor of size `0.8` raises the amplitude to roughly `1.35` and
creates killability around `0.62` while correcting back to exact multiplication.

Fixed-radius shell descent gives a stronger numerical barrier test.  After
curvature escapes over a wide range of sizes, constraining the algorithm to stay
on the corresponding shell and minimizing the selected amplitude still drives
it back toward approximately `1`, rather than below `1`.  This means the
previous return-to-one behaviour is not explained merely by retracing the
escape path.  It remains a numerical observation about the explored real
schoolbook-connected region, not a theorem about all rank-8 decompositions.

A checked shell-profile run gave:

| curvature escape size | shell radius | escaped amplitude | shell minimum amplitude |
|---:|---:|---:|---:|
| 0.8 | 0.8820 | 1.349218 | 1.000000032 |
| 1.2 | ~1.42 | 1.755447 | 1.000002494 |
| 1.6 | ~2.08 | 2.284468 | 1.000000000 |

The shell minimum is therefore numerically pinned extremely close to 1 in all
three checked cases.
