# String experiment: schoolbook -> Strassen

## Setup

Both endpoints are exact representations of the same `2 x 2` multiplication
tensor in an 8-channel CP parameterization. The Strassen solution uses seven
nonzero channels and one zero channel. Permutation and sign gauges are aligned
before forming the initial path.

The path uses 31 replicas and equal-arclength reparameterization. The potential
is the Frobenius norm of the multiplication-tensor residual.

## Observations

The aligned linear interpolation starts with a maximum residual of about
`7.56e-1`.

Ordinary normal string flow reduces that monotonically:

- after 5,000 steps: max residual about `1.19e-2`;
- after another 5,000 resumed steps: max residual about `8.71e-3`.

The residual was still falling, so this is **not evidence for a positive
asymptotic barrier**.

A line-searched Jacobian-whitened normal step (batched `J^+ F`, with the string
tangent removed) then reduced the maximum residual further to about `6.26e-3`
in five accepted steps before that particular preconditioned direction stalled.
This supports a hybrid strategy: robust ordinary string dynamics to change the
path shape, punctuated by Jacobian-whitened steps to cross stiff linear modes.

## The interesting structural result

The optimized path is highly organized. With the aligned channel labelling,
the amplitudes evolve approximately as:

- six channels move almost identically;
- one distinguished channel grows from `1` toward `2*sqrt(2)`;
- the eighth channel falls from `1` toward `0`.

At the exact Strassen endpoint the amplitudes are `[2*sqrt(2), 2, 2, 2, 2, 2,
2, 0]` up to the aligned ordering/sign gauge.

So the string is not wandering through arbitrary rank-8 decompositions. It has
found a low-dimensional, symmetry-respecting reaction coordinate that looks
qualitatively like the schoolbook-to-Strassen transition.

The remaining error is concentrated in the middle of this transition. At a
representative 10k-step path the maximum residual is around `8.7e-3` when the
dying channel has amplitude of order one, then the residual falls again as the
path approaches Strassen.

## Projection check

Independent Gauss-Newton projection drives most path replicas essentially back
to exact multiplication in a handful of iterations. A compact band near the
middle is harder: local correction needs order-one parameter displacement and
converges much more slowly. A separate trust-region least-squares probe from the
maximum-residual replica reduced residual from `1.19e-2` to `2.7e-8`, but moved
about `1.16` in parameter norm.

Interpretation: exact rank-8 solutions certainly exist away from both endpoints,
but the current string's middle section is not merely a tiny normal perturbation
of an obvious exact branch.

## What this does and does not say

It **does** say:

- the schoolbook and Strassen endpoints can be joined by a remarkably low-error
  continuous path found constructively;
- the path spontaneously reveals strong symmetry;
- Jacobian whitening is useful as a dynamical preconditioner.

It **does not** yet say:

- that the exact-algorithm set is disconnected;
- that a positive mountain-pass residual exists in the continuum limit;
- that the observed path is globally optimal.

The clean next tests are resolution scaling, alternating raw/whitened string
steps to a genuine plateau, and fitting a low-dimensional analytic ansatz to
the symmetry discovered by the numerical string.
