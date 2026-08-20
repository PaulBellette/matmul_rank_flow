# Analytic symmetry ansatz: schoolbook -> Strassen

## The numerical string was hiding an exact symmetry sector

The evolved 104-coordinate string obeys, to machine precision, fixed relations
between the three factor matrices.  One `4 x 8` matrix `U` determines the other
two by column permutations plus matrix transposition.  Inside the
schoolbook-connected part, `U` itself reduces to ten scalars:

```text
[ A  B  C  D -C -D  E  F ]
[ G  H  I  I  J  J -H -G ]
[-G -H  J  J  I  I  H  G ]
[ F  E -D -C  D  C  B  A ]
```

and the eight scalar multiplication amplitudes are

```text
[x, y, y, y, y, y, y, x].
```

Thus the relevant sector has only 12 scalar coordinates, rather than 104.

Substitution into the exact `2 x 2` multiplication tensor collapses the 64
tensor equations to **six distinct cubic equations**.  Fixing the unit-column
gauge contributes three quadratic equations.  The compact system therefore has
9 equations in 12 variables, and its generic numerical Jacobian has rank 9:
locally the exact algorithms form a **3-dimensional algebraic manifold**.

## The apparent string barrier is not a real residual barrier

Take the ordinary schoolbook point and the equal-split classical-Strassen
fusion point.  A straight line between them is not exact.  However, projecting
each point of that chord by minimum-normal-displacement Newton correction onto
the nine compact equations gives a smooth sequence of exact algorithms.

For 41 images in the supplied reference run:

- maximum compact-constraint residual: about `1e-12` or below;
- maximum full 64-entry multiplication-tensor residual: about `1e-12` or below;
- maximum neighbouring step in the 12-dimensional coordinates: about `5e-2`;
- maximum projection displacement from the raw chord: about `3.5e-1`.

So the `~6e-3` hump seen by the earlier 104-D string was a failure of the chosen
reaction coordinate / string dynamics, not evidence that exact multiplication
had to be violated.

## Closed-form rank-7/fusion family

At the fusion side, channels 0 and 7 become exactly the same rank-one tensor.
The remaining family has a closed one-parameter form.  Let

```text
C = -sin(theta)/sqrt(2)
D =  cos(theta)/sqrt(2)
B =  cos(theta)^2
E = -sin(theta)^2
H = -sin(theta) cos(theta)
```

and

```text
b(theta) = 2 / ((1 + sin(2 theta)) cos(2 theta)).
```

The factor matrix is

```text
[ r  B  C  D -C -D  E  r ]
[ 0  H  D  D -C -C -H  0 ]
[ 0 -H -C -C  D  D  H  0 ]
[ r  E -D -C  D  C  B  r ]
```

with `r = 1/sqrt(2)`, and amplitudes

```text
[p, b, b, b, b, b, b, 2*sqrt(2)-p].
```

This represents exact matrix multiplication for every nonsingular `theta` and
for any split `p`.  The reason `p` is free is simple: channels 0 and 7 are
identical, so only their sum matters.

At `theta = 0`:

- `p = sqrt(2)` is the equal-split fusion point;
- continuously move `p: sqrt(2) -> 2*sqrt(2)`;
- channel 7 dies exactly;
- the endpoint is the classical seven-product Strassen decomposition.

Hence we now have a constructive numerical exact homotopy

```text
schoolbook rank 8
    -> exact 3-D symmetry manifold
    -> equal-split duplicate-channel fusion
    -> exact duplicate-weight transfer
    -> Strassen rank 7 + zero channel.
```

The first stage is currently represented implicitly by nine low-degree
polynomial equations plus Newton projection.  The second stage is closed form.
The obvious next mathematical task is to parameterise the first 3-D algebraic
manifold explicitly (or find a particularly simple one-dimensional curve in
it), rather than treating the projection path as the final description.
