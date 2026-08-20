# Off-manifold basin-hop fallback

The autonomous controller previously terminated when repeated exact finite-shell
hops exhausted `hop_max_radius`.  The rank-24 exploratory run showed this was a
controller limitation rather than a tensor-search failure: the state remained
exact and locally mobile, but the shell corrector could no longer find another
nearby exact landing.

The controller now treats that event as a phase transition:

```text
CONTINUE_TO_WALL
      |
      v
     HOP -- exact shell exhausted --> OFF_MANIFOLD_HOP
      ^                                  |
      |                                  v
      +------ tunnel failed -------- soft shell relaxation
                                         |
                                         v
                                  exact basin recovery
                                         |
                                         v
                                CONTINUE_TO_WALL
```

`OFF_MANIFOLD_HOP` starts along a highly obstructed tangent direction, but does
not insist that the intermediate trajectory remain exact.  Instead it minimizes
physical-constraint error while maintaining a soft finite-radius shell around
the current exact solution.  Shell pressure is annealed, after which the normal
exact corrector is tried first on the shell and then, if necessary, without the
shell.

A landing is accepted only if:

- the full multiplication + unit-norm constraints return to `exact_tol`;
- all amplitudes remain below `coefficient_cap`;
- the solution remains a nontrivial finite distance from the starting basin.

Failed tunnels are nonterminal.  The controller resets to small exact hops with
a new random obstructed direction; `max_cycles` remains the global exploratory
budget.

Useful CLI controls:

```bash
--offhop-trials 3
--offhop-steps 350
--offhop-lr 0.015
```

The existing numerical SVD hardening and `latest_exact.pt` checkpoints remain
unchanged.

## Validation probe

A deliberately small endpoint-free probe from the exact rank-26 checkpoint used
one obstructed direction trial, radius 4, and only 100 Adam steps per soft-shell
stage.  The tunnel recovered a different exact rank-26 basin with:

```text
distance from home     4.44357
tensor residual        2.82e-15
max |amplitude|         3.63257
tangent nullity         135 -> 79
best death distance     2.71928 -> 2.07307
landing mechanism       tensor-only polish + canonical gauge
```

The tensor-only polish is used only when the full multiplication + unit-norm
corrector stalls.  Since `reconstruct()` normalizes each factor direction, raw
factor-column norms are gauge variables; after tensor equality is polished the
columns are normalized explicitly without changing the represented tensor.
