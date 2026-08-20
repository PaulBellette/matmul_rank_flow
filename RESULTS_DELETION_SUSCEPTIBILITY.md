# Deletion-susceptibility basin search

This iteration changes only the *global basin selection* logic of the autonomous
3x3 controller.  Local continuation, exact shell hops, the full delete probe,
and rank-drop acceptance are unchanged.

## Motivation

At rank 24 the previous controller could remain exact for many cycles while
cycling through states with similar local death distance `D = |a|/K` and the
same physical Jacobian nullity.  Off-manifold hops were finding different
coordinates, but `D` was not a sufficiently direct global signal for whether a
rank-23 basin was nearby.

The successful earlier blind rank drop had a more informative signature: when
a channel was clamped to zero, short residual relaxation suddenly changed from
an O(1) "missing product" residual to a much smaller residual before the exact
rank-(R-1) corrector succeeded.

## New score

For an exact rank-R tunnel landing, rank the locally most promising few
channels.  For each candidate r:

1. clamp `a_r = 0` immediately;
2. give the remaining variables a deliberately short bounded Adam relaxation;
3. record the best tensor residual.

Define

    E_delete(theta) = min_r E_r(theta)

where `E_r` is this *cheap* residual.  This helper cannot accept a rank drop; it
is only a basin-ranking diagnostic.  The existing full `DELETE_PROBE` and
`DROP` phases remain the only way to reduce rank.

The off-manifold hop now evaluates several tunnel landings and uses
`E_delete` as the primary branch score, with local death distance only as a
small tie-breaker.

## Basin archive

A small permutation/sign/gauge-insensitive fingerprint is stored for explored
exact states:

- sorted absolute amplitudes;
- eigenvalues of the Gram matrix of normalized rank-one channel tensors;
- maximum amplitude scale.

A tunnel landing that is extremely close to an archived fingerprint is rejected
as a duplicate.  This is only an exploratory anti-cycle heuristic, not a claim
that the fingerprint is a complete invariant.

## Important controller change

When a tunnel landing wins because channel r has the best deletion
susceptibility, the controller immediately performs the *full* delete probe on
that same channel.  It does not recompute the `a/K` winner first and wander away
from the basin selected by the global score.

## Validation

- full unit suite: 30/30 passing;
- compact end-to-end tunnel smoke produced an exact finite landing;
- the landing was scored by the new susceptibility probe;
- a `selected_summary` row identified the winning susceptibility channel.

The rank-24 run is the real exploratory test: does `E_delete` fall materially
under branched tunnel search, and does that lead to the final blind 24->23
transition?
