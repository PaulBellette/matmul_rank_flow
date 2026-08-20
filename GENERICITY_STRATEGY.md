# Rank-generic basin search by Jacobian genericisation

The controller now treats rank reduction as a two-timescale process that is the
same at every current rank `R`:

1. **Local exact motion:** exploit a currently killable amplitude until its
   continuation reaches a wall/fold.
2. **Global basin motion:** prefer exact or off-manifold tunnel landings that
   open previously-null physical Jacobian modes.
3. **Rank-drop switch:** deletion susceptibility is a secondary global signal;
   if a landing is already close to an `(R-1)` basin, its short delete probe can
   override genericisation naturally.
4. **Drop and reset:** after an accepted rank drop, infer a new spectral scale
   from the new rank-`R-1` Jacobian and repeat. No desired nullity is encoded.

## Soft effective nullity

For the physical constraint Jacobian `J`, the current state defines a spectral
scale `tau`. Using the controller's existing numerical-rank cutoff, let

- `sigma+` = smallest resolved positive singular value;
- `sigma0` = largest numerical-null singular value.

When both exist,

```text
tau = sqrt(sigma+ * sigma0)
```

A candidate basin is always evaluated using the **home state's same `tau`**:

```text
N_tau(J) = n_parameters - sum_i sigma_i^2 / (sigma_i^2 + tau^2)
```

This is a smooth version of tangent nullity. A zero/soft mode that opens while
moving to another basin continuously lowers `N_tau`.

The scale is inferred afresh after every rank drop, so there is no hard-coded
`nullity=45`, `rank=24`, or known rank-23 endpoint in the objective.

## Global branch score

For a candidate tunnel landing, lower is better:

```text
score = 6 * N_tau(candidate) / N_tau(home)
      + log10(1 + E_delete / 0.1)
      + 0.02 * log(1 + death_distance)
      + 0.05 * max_amplitude / coefficient_cap
```

`E_delete` is the existing cheap deletion-susceptibility probe. The weights are
exploratory, but their intent is simple:

- while deletion is hard (`E_delete ~ 1`), opening tangent constraints is the
  main global objective;
- if a candidate becomes genuinely close to a lower-rank basin
  (`E_delete << 0.1`), the logarithmic deletion term takes over automatically.

Ordinary exact shell hops use the same genericity score but omit the expensive
`E_delete` term.

## Sanity check

Using the existing rank-26 checkpoints:

- embedded-Strassen start: hard/soft nullity ~ `135 / 135`;
- an endpoint-free basin-hop checkpoint from earlier exploration: hard nullity
  `84`, with soft nullity also dramatically reduced at the home spectral scale.

A fresh two-trial exact-hop validation from the rank-26 start selected a landing
whose hard nullity fell to `53`. At the *home* smoothing scale its soft nullity
was `~74.79`, correctly recognizing that some newly-opened modes were still
very soft rather than pretending all 82 hard-rank gains were equally stiff.

That conservative behaviour is intentional.
