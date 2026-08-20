# Autonomous collision search

This stage removes the hard-coded Strassen collision pair from the search.
The algorithm starts only from the exact 8-product schoolbook decomposition.

## Local score

For a channel pair `(r,s)` and factor-sign pattern `sigma` with
`prod(sigma)=+1`, define the constructive collision coordinate

\[
f_{rs}^{\sigma}
=\frac{1}{3}\left(
\sigma_u\langle u_r,u_s\rangle+
\sigma_v\langle v_r,v_s\rangle+
\sigma_w\langle w_r,w_s\rangle
\right).
\]

`f=1` means the two normalized rank-one tensors coincide with constructive
(overall positive) sign and can be fused by adding their amplitudes.

The useful second-order object is **not** the ambient Hessian.  If the physical
exact-algorithm constraints are `G(theta)=0`, then at a constrained stationary
point

\[
K_f=N^T\left(H_f-\sum_i\lambda_i H_{G_i}\right)N,
\qquad J^T\lambda=\nabla f,
\]

where the columns of `N` span `ker J`.

We further restrict this operator to the nullspace of the sum of the two
channel-amplitude curvature operators.  Thus a pair only scores highly if its
factors can begin to collide **without making the candidate amplitudes grow at
second order**.

## Schoolbook scan

The scan of all 28 channel pairs finds a sharp split.

The unique best symmetry class is

- `(0,7)`
- `(1,6)`
- `(2,5)`
- `(3,4)`

For all four:

- initial factor cosines are `(0,0,0)`;
- tangent collision gradient is zero to numerical precision;
- pair-amplitude-flat tangent dimension is 18;
- best constrained collision curvature is `1/3`.

The next class has curvature `1/6`.

Thus the local geometry autonomously identifies the four opposite-corner
schoolbook pairs, exactly the symmetry class containing the known `(0,7)`
closed-form fusion path.

## Autonomous flow

The search then:

1. selects one of the four tied pairs using only a random tie-break seed;
2. takes the top constrained-curvature eigenvector;
3. constructs the second-order exact predictor

   \[
   \theta(s)=\theta_0+s q+\tfrac12s^2 z,
   \qquad
   z=-J^+D^2G[q,q];
   \]

4. corrects back to the exact-algorithm manifold;
5. performs tangent-projected ascent of the collision coordinate, with a small
   penalty against unnecessary growth of the two candidate amplitudes;
6. when the factor cosines are nearly one, adds exact factor-coincidence
   constraints and Newton-corrects to an exact collision;
7. fuses the duplicate terms.

## Representative result: seed 0

No pair is supplied to the solver.

The four tied candidates were

```text
(2,5), (3,4), (0,7), (1,6)
```

The seeded tie-break chose `(1,6)` with signs `(-,+,-)`.

The collision coordinate evolved

```text
0.1060327
0.3883881
0.6453929
0.8692488
0.9818389
0.9974813
0.9999748
0.9999994
```

while exact tensor/unit-norm constraints remained at numerical precision.

Immediately before the final exact collision correction the pair amplitudes
were approximately

```text
1.41421297, 1.41421296
```

The collision corrector converged to residual

```text
9.15e-16
```

with amplitudes

```text
1.4142135679, 1.4142135568
```

and fusing the two terms produced a 7-product decomposition with full tensor
residual

```text
1.15e-15
```

## Symmetry checks

Seeds 0--4 all independently selected a member of the four-pair best symmetry
class and all reached an exact rank-7 fused decomposition.  Separate directed
testing also verifies the `(0,7)` member.

## Interpretation

The constructive mechanism is now discoverable from local dynamics alone:

```text
schoolbook
  -> identify second-order amplitude-flat collision pair
  -> curvature-guided exact escape
  -> exact-manifold collision ascent
  -> duplicate rank-one terms
  -> fuse
  -> rank 7
```

The important lesson is that **rank reduction is better represented as a
collision problem than as channel death**.  The schoolbook decomposition is
locally stable against amplitude deletion, but its constrained curvature
contains directions that cause a symmetry-selected pair of rank-one channels
to approach one another while their amplitudes remain initially flat.

This is the formulation worth trying on larger matrix-multiplication tensors.
