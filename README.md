# Matrix multiplication rank-flow toy

This is a deliberately small experiment in treating matrix-multiplication
algorithm search as a continuous optimization problem.

For `n=2`, the matrix multiplication tensor is a `4 x 4 x 4` tensor with eight
nonzero entries.  We represent it as

\[
\hat T = \sum_{r=1}^R a_r u_r \otimes v_r \otimes w_r.
\]

The factor directions are normalized to unit length, which removes the simplest
continuous CP scaling gauge, and amplitudes are bounded.

There are three experiments.

## 1. Fixed-rank continuous baseline

Can ordinary gradient optimization discover a rank-7 representation from
random initialization?

```bash
uv run python rankflow.py --mode fit --n 2 --rank 7 --seeds 10
```

This is **not** rank selection. It is a sanity check that the continuous
parameterization and optimizer can reach the known low-rank stratum.

## 2. Soft rank flow

Start overparameterized and add a nonconvex, smooth approximation to the number
of active channels:

\[
L =
\beta\|T-\hat T\|_F^2
+\lambda(t)\sum_r(a_r^2+\epsilon(t)^2)^{p/2},
\qquad 0<p\ll1.
\]

```bash
uv run python rankflow.py \
  --mode soft \
  --n 2 \
  --rank 8 \
  --soft-init random \
  --steps 4000 \
  --lambda-max 0.15 \
  --p 0.12
```

`lambda` rises and `epsilon` falls.  If this works as hoped, an amplitude should
collapse while the tensor residual stays very small.

Try `--soft-init naive` as the deliberately harder test.

## 3. Forced channel death / homotopy

Start from the schoolbook 8-product decomposition and externally gate one
channel from `1 -> 0`.  The other channels are allowed to move continuously to
try to preserve exact multiplication.

```bash
uv run python rankflow.py \
  --mode kill \
  --n 2 \
  --kill-channel 0 \
  --steps 6000
```

A successful continuous route to rank 7 would keep the residual near zero all
the way to gate zero.

Run all three probes:

```bash
uv run python rankflow.py --mode demo
```

Outputs go to `runs/rankflow/`:

- per-step CSV trajectories,
- model state dictionaries,
- `summary.json`.

## What to watch

The useful measurements are:

1. tensor residual,
2. active channel count,
3. channel amplitudes,
4. the point at which a forced-death trajectory loses feasibility.

A particularly interesting failure is

- residual ~ 0 for much of the path,
- then a sharp rise as a channel approaches zero.

That suggests the current exact decomposition sits in a basin/stratum that does
not smoothly reach the lower-rank decomposition under this particular
homotopy.  That does **not** rule out other continuous paths; it tells us we
need a better escape/search mechanism.

## Obvious next mutations

If the simple flow stalls:

- inject tangent-space noise when progress stalls;
- try all candidate dying channels in parallel;
- allow pairwise "merge/split" moves between rank-one terms;
- optimize a population of decompositions and exchange terms;
- use an augmented-Lagrangian equality constraint rather than a fixed residual
  weight;
- detect border-rank behaviour by tracking factor/amplitude blow-up;
- after finding a low-residual solution, exactify simple coefficients with
  rational/integer recognition.

The first question is intentionally much smaller: **does continuous rank death
show a reproducible geometry at all?**

---

# Jacobian geometry / exact-manifold continuation

`geometry_flow.py` is the dynamics-oriented follow-up.  Rather than balancing a
residual penalty against a sparsity penalty, it studies the local manifold of
**exact** matrix-multiplication algorithms.

At an exact decomposition define

\[
F(\theta)=\operatorname{vec}(\hat T(\theta)-T),\qquad J=DF(\theta).
\]

Then `ker(J)` is the first-order tangent space of exact algorithms.  The script
uses an SVD of `J` to measure:

- numerical rank and nullity;
- smallest positive singular value and condition number;
- per-channel **killability**
  \(\|P_{\ker J}e_r\|\), where `e_r` changes only channel `r`'s amplitude;
- the minimum tangent-motion cost `1 / killability` required to change that
  amplitude at unit rate.

## Inspect the schoolbook point

```bash
uv run python geometry_flow.py --mode geometry
```

Outputs:

- `geometry.json`
- `geometry_spectrum.csv`
- `geometry_killability.csv`

## Follow the exact manifold toward channel death

```bash
uv run python geometry_flow.py \
  --mode manifold \
  --target-channel -1 \
  --amp-step 0.025
```

`-1` picks the channel with greatest local killability.

Each predictor step is the minimum-norm tangent direction satisfying

\[
J\,d\theta=0,\qquad da_r/ds=-\operatorname{sign}(a_r).
\]

The corrector is a truncated-SVD Gauss--Newton projection back onto `F=0`, with
the selected amplitude frozen at its newly predicted value.  The amplitude
step is automatically halved if the exact corrector cannot converge.

Watch `manifold_flow.csv` for:

- `killability -> 0`: the selected amplitude has become a bad continuation
  coordinate / a possible fold;
- `unit_change_cost -> infinity`: tangent motion required to keep decreasing
  the channel is blowing up;
- `sigma_min_positive -> 0`: another Jacobian mode is softening;
- `corrector_norm / predictor_norm`: how nonlinear the exact manifold has
  become locally.

A clean success is target amplitude `-> 0` with residual near machine precision.
A clean failure is also useful: it should now tell us *geometrically* why the
path stopped rather than merely reporting a large loss.

## Tangent kick

The exact schoolbook point is very symmetric.  To move sideways on the exact
algorithm manifold before choosing/decreasing a channel:

```bash
uv run python geometry_flow.py --mode manifold --kick 0.05 --seed 3
```

The kick is projected into `ker(J)` and then Gauss--Newton corrected back to an
exact decomposition before rank-death continuation begins.

## Curvature escape search

At the exact schoolbook point the amplitude directions can be completely absent
from `ker(J)`: there may be sideways exact motions but no first-order way to
reduce a product.  `--escape-size/--escape-trials` probes this second-order
geometry cheaply:

```bash
uv run python geometry_flow.py \
  --mode manifold \
  --escape-size 0.8 \
  --escape-trials 60 \
  --seed 1
```

It samples tangent directions, removes the useless radial gauge of normalized
CP columns, takes a finite sideways step, Gauss--Newton corrects back to the
exact manifold, and keeps the exact candidate with the largest channel
killability.  The trial table is written to `escape_trials.csv`.

This is useful when the symmetric starting point is singular: amplitude motion
may be zero at first order but become available after finite motion along the
exact manifold.

---

# Second-order curvature flow

`curvature_flow.py` removes the factor-normalisation gauge explicitly and studies
second-order motion of the exact-algorithm manifold.

The physical constraints are

\[
G(\theta)=0,
\]

where `G` contains both exact tensor equality and unit-norm constraints on every
CP factor direction.  If `q` is tangent (`J q = 0`), a second-order exact path
has normal acceleration `z` satisfying

\[
Jz=-D^2G[q,q].
\]

For channel amplitude `a_r`, the script constructs the quadratic curvature
operator `K_r` such that

\[
\frac{d^2 a_r}{ds^2}=q^T K_r q.
\]

Inspect it with

```bash
uv run python curvature_flow.py --mode curvature
```

At the `2 x 2` schoolbook point the gauge-fixed Jacobian is `88 x 104`, rank
`80`, leaving a 24-dimensional physical tangent space.  Numerically, each of
the eight channel curvature operators is positive semidefinite, rank 3, with
three nonzero eigenvalues equal to 1.  Thus every schoolbook amplitude `a_r=1`
is a second-order local floor on the exact manifold.

## Deterministic curvature escape

Instead of sampling random tangent directions, use the top eigenvector of a
channel's curvature operator and a second-order predictor:

```bash
uv run python curvature_flow.py --mode escape --channel 0 --size 0.8
```

This deliberately moves in the direction that creates amplitude motion fastest,
then corrects back to the exact manifold.

## Fixed-radius shell profile

To distinguish "the flow merely retraced the escape" from a genuine local
barrier, constrain the algorithm to remain a fixed parameter-space distance
from the schoolbook point and minimize one channel amplitude around that shell:

```bash
uv run python curvature_flow.py --mode profile
```

The output `shell_profile.csv` reports the minimum channel amplitude found at
several shell radii.  This is a local geometric diagnostic, not a proof of a
global invariant or disconnected components.

## String / mountain-pass experiment

`string_flow.py` connects two exact endpoints in the rank-8 parameter space:

1. the exact schoolbook 8-product algorithm;
2. the exact classical Strassen 7-product algorithm with a zero eighth channel.

The Strassen endpoint is aligned to the schoolbook endpoint over CP permutation
and sign gauges before interpolation. Interior replicas descend the tensor
residual **normal to the current string**, followed by equal-arclength
reparameterization.

```bash
python string_flow.py --images 31 --steps 5000 --lr 0.08 --reparam-every 5
```

The linearized dynamics near an exact decomposition are stiff because vanilla
gradient flow contains `J.T @ J`. The optional whitened step solves
`J delta ~= -F` by batched SVD, removes the along-string component, and performs
a line search on the maximum path residual:

```bash
python string_flow.py \
  --resume runs/string/string_points.pt \
  --steps 2000 \
  --whiten-every 200
```

This is intentionally a geometry probe, not a claim that a non-zero sampled
barrier proves disconnected exact components. Increase resolution, run longer,
and inspect conditioning before interpreting a plateau as a genuine barrier.

## Exact low-dimensional symmetry ansatz

`analytic_ansatz.py` is the next step after the string experiment.  It reduces
the observed schoolbook-to-Strassen reaction coordinate to 12 scalar variables
and nine polynomial constraints, constructs a smooth exact schoolbook-to-fusion
path, and supplies a closed-form rank-7/fusion family.

```bash
uv run python analytic_ansatz.py --mode demo --out runs/ansatz
```

For the closed-form rank-7 family alone:

```bash
uv run python analytic_ansatz.py --mode rank7 --theta 0.1
```

See `RESULTS_ANSATZ.md` for the derivation and interpretation.

## Closed-form schoolbook -> Strassen path

The numerical manifold projection can now be replaced by an explicit
one-parameter exact branch.  See `RESULTS_CLOSED_FORM.md` and run:

```bash
uv run python closed_form_homotopy.py --mode demo --out runs/closed_form
uv run python symbolic_closed_form.py
```

The key mechanism is **channel collision rather than amplitude death**: two
rank-one multiplication channels continuously become identical, after which
their weights can be fused exactly.

## Autonomous collision search

`collision_search.py` removes the hard-coded Strassen channel pair.  It scans
all schoolbook channel pairs using constrained second-order collision curvature,
requires constructive sign and zero second-order growth of the candidate pair
amplitudes, follows the selected exact-manifold direction, and fuses the first
collision it finds.

```bash
python collision_search.py --mode scan --out runs/collision_scan
python collision_search.py --mode search --seed 0 --out runs/collision_search
```

For `2x2`, the scan autonomously identifies the four symmetry-equivalent
opposite-corner pairs `(0,7)`, `(1,6)`, `(2,5)`, `(3,4)`.  A representative
search reaches an exact rank-7 fused decomposition with tensor residual around
`1e-15`.  See `RESULTS_COLLISION_SEARCH.md`.

## 3x3 scaling experiment

`collision_search_3x3.py` scales the collision operator to schoolbook `3x3`
multiplication without brute-forcing 351 expensive Hessians.  Schoolbook index
symmetry reduces the pair scan to seven orbit representatives.

Run:

```bash
uv run python collision_search_3x3.py --mode demo --seed 0 --out runs/collision_3x3
```

Outputs include:

- `orbit_scan.csv` -- constrained curvature for the seven schoolbook pair orbits;
- `embedded_homotopy.csv` -- exact full-3x3 path for the selected local collision;
- `rank26.pt` -- exact 26-product decomposition after fusion;
- `rank26_first_order_pair_mobility.csv` -- cheap diagnostic for the next stage;
- `summary.json` -- compact run summary.

See `RESULTS_3X3.md` for the interpretation.

## 3x3 endpoint-guided cascade

See `RESULTS_GUIDED_CASCADE_3X3.md` and `guided_cascade_3x3.py` for the later
`27 -> 26 -> 25 -> 24 -> 23` experiment.  The `27 -> 26` collision is locally
discovered; the later stages use an independent exact rank-23 decomposition as
a **global guide**, so they should not be described as autonomous discovery.

```bash
uv run python guided_cascade_3x3.py --mode verify
uv run python guided_cascade_3x3.py --mode last-drop
```

## Endpoint-free hybrid state machine (exploratory)

`autonomous_state_machine_3x3.py` packages the manual exploratory recipe into a
hybrid controller.  It does **not** use the known rank-23 endpoint.

The state machine alternates:

1. `CONTINUE_TO_WALL` — use the projected amplitude gradient to follow an exact
   branch until local killability collapses or the familiar `|a| ~= 1` wall is
   reached;
2. `HOP` — select tangent directions with large second-order non-integrability
   and solve for a different exact algorithm on a finite-radius shell;
3. `DELETE_PROBE` — temporarily leave the exact manifold and clamp a promising
   channel through the wall, with a hard finite-coefficient guard;
4. `DROP` — only accept the event if deleting that channel can be corrected to
   an exact rank-`R-1` decomposition.

Run an exploratory blind controller from the exact autonomous rank-26 state:

```bash
uv run python autonomous_state_machine_3x3.py \
  --start reference_guided_cascade/rank26.pt \
  --goal-rank 23 \
  --out runs/autonomous_state_machine
```

For a very small wiring check:

```bash
uv run python autonomous_state_machine_3x3.py --smoke --out runs/state_machine_smoke
```

The controller is intentionally still exploratory: it records every continuation,
shell hop, deletion probe and accepted rank drop so the hybrid dynamics can be
inspected before doing seed/invariance sweeps.

## Generic basin objective

`autonomous_state_machine_3x3.py` now uses a rank-generic Jacobian
**genericisation** objective for global basin changes. It infers a soft-nullity
scale from the current physical Jacobian spectrum, rewards landings that open
excess tangent modes, and uses cheap deletion susceptibility as a secondary
signal that automatically dominates when an `(R-1)` basin becomes genuinely
nearby. See `GENERICITY_STRATEGY.md`.

## Pareto beam controller

`autonomous_state_machine_3x3.py` now defaults to a tiny Pareto beam over exact
algorithm basins rather than a single greedy basin walk. See
`PARETO_BEAM_STRATEGY.md` for the rationale and metrics. Use
`--search-mode greedy` to reproduce the previous controller.

## Specialist-scheduled Pareto beam

The default beam controller now expands frontier states according to *why* they
are useful (genericity, deletion susceptibility, or local death distance), not
only by a balanced rank score.  It also tracks lineage/expansion counts and
polishes retained exact basins before spectral comparison.  See
`SPECIALIST_PARETO_BEAM.md`.

## Freeze and exactify a blind rank-23 run

Once an endpoint-free beam run reaches rank 23, copy/promote the successful
checkpoint into `results/` and exactify it independently of the search code:

```bash
python3 exactify_rank23.py \
  results/blind_rank23/final.pt \
  --out results/blind_rank23/exactify \
  --compare-reference
```

The exactifier:

1. tensor-polishes the numerical checkpoint with a standalone Gauss-Newton
   corrector;
2. canonicalises the per-channel CP scaling gauge;
3. tries rational and small-radical coefficient recognition;
4. verifies recognised candidates against all 729 Brent identities exactly
   with SymPy;
5. optionally reports a conservative direct match against the published
   ternary rank-23 certificate (channel permutation + channel scaling only).

A failed direct match is not evidence of a new equivalence class: the script
currently does **not** search the full GL(3) isotropy group of matrix
multiplication.

Transient exploratory output belongs in `runs/`; promote anything worth
keeping/reproducing into `results/`.  `.gitignore` intentionally ignores
`runs/` but does not ignore `results/` or reference checkpoints.

## Exact certificate for the blind rank-23 family

The first endpoint-free specialist-beam success has now been taken beyond a
numerical residual.  A well-conditioned incidence-derived `GL(3)^3` gauge
exposes a sparse 170-variable Brent system with a 21-dimensional local tangent
family.  Moving only about `5.84e-3` in those reduced coordinates to lock 21
simple rational parameters, followed by high-precision refinement, gives an
exact rank-23 representative over

```text
Q(sqrt(85213608769)).
```

The bundled exact certificate has 594 rational coefficients and 50 genuinely
quadratic coefficients, and all 729 Brent identities vanish symbolically.

See `EXACT_RANK23_RESULT.md` for the precise claim and reproduction commands.
