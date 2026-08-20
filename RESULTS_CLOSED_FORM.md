# Closed-form schoolbook -> Strassen homotopy

## 1. The projected 3-D manifold contains a much simpler 1-D exact branch

Inside the 12-variable symmetry ansatz, impose

```text
B = 1
C = E = G = H = J = 0.
```

The nine compact equations then reduce to five nontrivial conditions:

```text
x (A^3 + F^3) = 1
x A F (A + F) = y D^2
y I^2 = 1
A^2 + F^2 = 1
D^2 + I^2 = 1.
```

Let

```text
p = A F,       0 <= p <= 1/2.
```

Because `A^2+F^2=1`,

```text
(A+F)^2 = 1 + 2p
A^3+F^3 = (A+F)(1-p).
```

Choose the positive branch and set

```text
A = (sqrt(1+2p) + sqrt(1-2p))/2
F = (sqrt(1+2p) - sqrt(1-2p))/2
D = sqrt(p)
I = sqrt(1-p)
x = 1 / ((1-p) sqrt(1+2p))
y = 1 / (1-p).
```

Then every compact equation vanishes identically.  `symbolic_closed_form.py`
asks SymPy to verify this directly.

- `p=0` is exactly the schoolbook eight-product decomposition.
- `p=1/2` is exactly the equal-split fusion point.

No numerical projection is required.

## 2. A smooth dynamical coordinate

Use

```text
p = s^2,        0 <= s <= 1/sqrt(2).
```

Near schoolbook,

```text
D = s
F = s^2 + O(s^6)
I = 1 - s^2/2 + O(s^4)
y = 1 + s^2 + O(s^4)
x = 1 + 3 s^4/2 + O(s^6).
```

This explains the earlier Jacobian/curvature observations.  The exact path
leaves schoolbook primarily by rotating/mixing factor directions.  No channel
amplitude decreases.  The six middle amplitudes rise only at second order; the
two channels that eventually fuse are even flatter, changing only at fourth
order in this particular tangent direction.

## 3. Rank reduction occurs by collision, not by shrinkage

Channels 0 and 7 have unit-normalized factor vectors.  Along the branch,

```text
<u0,u7> = <v0,v7> = <w0,w7> = 2 p = 2 s^2.
```

Thus they start orthogonal at schoolbook and monotonically become identical at
`p=1/2`.  The cosine similarity of the normalized rank-one tensors is

```text
(2p)^3.
```

Their amplitudes do **not** shrink:

```text
x: 1 -> sqrt(2).
```

Meanwhile the other six amplitudes move

```text
y: 1 -> 2.
```

At the endpoint channels 0 and 7 are literally the same rank-one tensor with
weights `sqrt(2), sqrt(2)`.  Only then is rank reduction trivial: transfer
weight continuously

```text
(sqrt(2), sqrt(2)) -> (2 sqrt(2), 0).
```

The represented matrix-multiplication tensor is unchanged throughout the
transfer, and the final decomposition is Strassen with seven active products.

This changes the useful search primitive.  The successful route is not

```text
make one amplitude vanish while preserving exactness.
```

It is

```text
move two rank-one channels together while preserving exactness
    -> make them coincide
    -> fuse their weights.
```

For constructive searches at larger sizes, pairwise projective collision of
rank-one summands may therefore be a more useful continuous objective than
sparsity of channel amplitudes.

## 4. The eight rank-one factors along the branch

With flattening `[11,12,21,22]`, write the four surviving scalar factor
parameters as `A,F,D,I`.  The `U` factor matrix is

```text
[ A  1  0  D  0 -D  0  F ]
[ 0  0  I  I  0  0  0  0 ]
[ 0  0  0  0  I  I  0  0 ]
[ F  0 -D  0  D  0  1  A ]
```

and the fixed cyclic symmetry generates `V` and `W`.  The amplitudes are

```text
[x, y, y, y, y, y, y, x].
```

At `p=0` these are precisely the eight schoolbook scalar products.  At
`p=1/2` the first and last factor triples coincide.

## 5. Reproduce

```bash
uv run python closed_form_homotopy.py --mode demo --out runs/closed_form
uv run python symbolic_closed_form.py
uv run pytest -q
```

`closed_form_homotopy.py` also exports the individual eight linear forms at any
point on the branch.

## 6. A much simpler gauge: the whole proof is one scalar relation

The normalized tensor coordinates obscure an especially simple algorithmic
form.  Define a mixing parameter `t in [0,1]` and a companion parameter `z`
by

```text
t^2 (z^2 - z + 1) = z.
```

The branch that starts at schoolbook is

```text
z(t) = 2 t^2 / (1+t^2 + sqrt((1-t^2)(1+3t^2))).
```

Thus `z(0)=0` and `z(1)=1`.  Put `lambda=1/(1+z^3)`.
For

```text
A = [a b; c d],   B = [e f; g h],
```

use the eight products

```text
m0 = lambda (a + z d)(e + z h)
m1 = a(f - t h)
m2 = (b - t d)(g + t h)
m3 = (t a + b)h
m4 = (c + t d)e
m5 = (-t a + c)(t e + f)
m6 = d(-t e + g)
m7 = lambda (z a + d)(z e + h)
```

and recombine

```text
C11 = m0 + m2 - t m3 + t m6 + z m7
C12 = m1 + m3
C21 = m4 + m6
C22 = z m0 + t m1 - t m4 + m5 + m7.
```

A direct SymPy expansion gives

```text
C12_error = 0
C21_error = 0

C11_error = -(a h + d e + d h) R/(z^2-z+1)
C22_error = -(a e + a h + d e) R/(z^2-z+1)
```

where

```text
R = t^2(z^2-z+1)-z.
```

So **the entire correctness proof reduces to the single scalar condition
`R=0`**.

At `t=z=0`, these are exactly the eight schoolbook products.  At `t=z=1`,
`m0` and `m7` are identical half-weight copies of

```text
(a+d)(e+h),
```

with the corresponding diagonal output contribution; adding them produces the
usual Strassen diagonal product.  The other six products are already the six
remaining Strassen-like products.

This is probably the cleanest description of the seam discovered by the
continuous dynamics.
