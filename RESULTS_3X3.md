# 3x3 collision-geometry results

## Question

Can the collision-geometry rank-reduction operator discovered for `2x2`
identify a useful reduction in the `3x3` schoolbook algorithm without being
told where a Strassen block lives?

## Schoolbook geometry

For `n=3` the physical schoolbook parameterization has

- 27 multiplication channels,
- 756 parameters,
- 810 physical constraints (729 tensor entries + 81 unit-norm gauges),
- Jacobian rank 594,
- exact tangent dimension 162.

There are `C(27,2)=351` channel pairs.  At schoolbook, independent
permutations of the three schoolbook indices `(i,j,k)` reduce these to only
seven orbit types, described by whether each index agrees or differs.

The constrained collision-curvature scan gives:

| mask | pair count | collision curvature |
|---|---:|---:|
| `111` | 108 | `1/3` |
| `001` | 27 | `1/6` |
| `010` | 27 | `1/6` |
| `100` | 27 | `1/6` |
| `011` | 54 | `1/6` |
| `101` | 54 | `1/6` |
| `110` | 54 | `1/6` |

The important qualitative result is exact:

> The unique best orbit is `111`: the two schoolbook channels differ in all
> three indices.

There are 108 such pairs.  Each is an opposite-corner pair of one of the
`C(3,2)^3 = 27` embedded `2x2x2` cubes, with four opposite pairs per cube.

Thus the local dynamics independently rediscovers where a `2x2` Strassen
reduction can live inside `3x3` multiplication.

## Exact 27 -> 26 reduction

For seed 0 the tie-breaker selected global channels `(6,11)`:

- channel 6 = `(i,j,k) = (0,2,0)`
- channel 11 = `(i,j,k) = (1,0,2)`

These determine the oriented index cube

- `I = (0,1)`
- `J = (2,0)`
- `K = (0,2)`.

The eight schoolbook products in that cube form an exact `2x2` matrix
multiplication subproblem.  Embedding the previously discovered exact
schoolbook-to-collision homotopy into this cube while leaving the other 19
schoolbook products untouched gives a full `3x3` homotopy with sampled maximum
residual

    9.36e-16

At the endpoint one local channel has zero amplitude.  Dropping it leaves 26
multiplication channels with full `3x3` tensor residual

    6.28e-16

So the first autonomous `3x3` rank reduction is exact:

    27 -> 26.

## After the first fusion

The rank-26 point is no longer schoolbook-symmetric.  A cheap first-order
projected-gradient scan of its 325 channel pairs has a much less clean
spectrum: 23 distinct `(mobility, current-collision)` classes.  The largest
projected collision-gradient norm is `2/3`, attained by 45 pairs, all between
schoolbook channels left outside the first reduced cube.

That is deliberately *not* yet treated as the next fusion signal.  In the
`2x2` work, large first-order collision mobility alone admitted cancellation /
border-rank-like false positives.  The next stage should therefore carry the
same safeguards forward: amplitude curvature, conditioning/blow-up penalties,
and exact-manifold correction.

## Interpretation

The first `3x3` result is less mysterious than the original `2x2` discovery,
but it is exactly what a reusable rank-reduction primitive should do:

1. inspect the local exact-algorithm geometry;
2. identify an equivalence class of promising channel collisions;
3. localize the smallest closed subproblem associated with the collision;
4. apply the exact collision/fusion operator there;
5. return a lower-rank exact algorithm.

For schoolbook `3x3`, step 3 discovers an embedded `2x2x2` cube and step 4 is
our already-learned `2x2` primitive.

The genuinely new problem begins at rank 26, after the obvious schoolbook
symmetry has been broken.
