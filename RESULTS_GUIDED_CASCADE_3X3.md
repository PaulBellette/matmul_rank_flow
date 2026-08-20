# Endpoint-guided 3x3 cascade: 27 -> 23

## What changed

The autonomous collision geometry gave an exact `27 -> 26` reduction by
embedding the discovered 2x2 channel-collision homotopy in one 2x2x2 cube.
After that, local pair-collision and channel-dependence objectives repeatedly
found coefficient-blow-up directions.

A finite-amplitude dependence search reached a stationary/near-singular region,
but a partial amplitude lock showed that the local dependence objective had no
useful second-order descent left.  At that point we deliberately switched from
**autonomous search** to **endpoint-guided geometry**.

An independent exact rank-23 ternary scheme was embedded as rank 26 with three
zero channels and used only as the far endpoint of a string.  This is therefore
not a claim that the method autonomously rediscovered a 3x3 rank-23 scheme.

## Exact reference endpoint

`rank23_reference.py` reconstructs the ternary rank-23 coefficient arrays used
as the guide.  In our tensor convention its residual is about `3e-15`.

## Rank-26 -> rank-25

The raw aligned rank-26/rank-23 straight path had maximum residual about `7.7`.
Unconstrained string flow initially reduced this rapidly but then escaped by
coefficient blow-up.  Adding a finite-amplitude trust region changed the
behaviour completely:

- max residual after the first guarded stage: `0.3610`
- after another 3000 steps: `0.2272`
- after another 3000 steps: `0.1665`
- largest amplitude decreased from about `9.91` to `9.37`

The endpoint-zero channels were 0, 10 and 12.  At image 20, channel 0 had
already fallen to about `0.222` and the tensor residual was only `0.0270`.
Projecting that replica toward the exact rank-26 manifold spontaneously drove
channel 0 to about `5e-9`, with max amplitude about `7.00`.  Dropping it and
polishing in rank 25 produced the supplied rank-25 checkpoint, residual about
`3.4e-11` in float64.

This is the first reduction after the embedded Strassen block that genuinely
requires interaction with the rest of the 3x3 decomposition.

## Rank-25 -> rank-24

After removing the first dead channel, the two remaining endpoint-zero channels
were indices 9 and 11 in rank-25 coordinates.

A short guarded string to the same rank-23 endpoint was much easier:

- initial max residual: `0.344`
- 100 steps: `0.0583`
- 300: `0.0319`
- 600: `0.0232`
- 1000: `0.0182`
- 1500: `0.0148`
- 2000: `0.01249`

Near the endpoint, channel 11 was about `0.001`.  Dropping it gave a rank-24
residual `2.27e-3`; four Gauss-Newton corrections reduced this to
`6.19e-12`, with max amplitude about `8.82`.

## Rank-24 -> rank-23

The final rank-24 checkpoint has only one endpoint-zero channel left, index 9,
with amplitude `0.0456108`.

At this stage local continuation works directly.  Constraining that amplitude
through nine evenly spaced targets down to zero required only about two Newton
corrections per step.  Max amplitude stayed essentially fixed around `8.82`.
At zero amplitude:

- rank-24 tensor residual: `4.59e-15`
- after deleting channel 9, rank-23 residual: `4.63e-15`

So the complete numerical finite-coefficient cascade is

```text
27 --local cube collision--> 26
26 --global finite reorganisation--> 25
25 --short guided reorganisation--> 24
24 --regular amplitude continuation--> 23
```

## The conceptual lesson

The 2x2 mechanism (rank loss by pair collision) was too narrow as a universal
objective.  At 3x3, after the first embedded Strassen reduction, local
collision/dependence objectives preferentially point toward border-rank-style
coefficient blow-up.

The useful rank-26 -> rank-25 transition is instead a **finite reorganisation
that changes which amplitude directions are dynamically accessible**.  Once
that transition is crossed, ordinary amplitude-death coordinates re-emerge and
the remaining reductions become progressively easier.

This suggests a search strategy with two scales:

1. local curvature/dynamical invariants to detect when a rank-drop direction is
   available;
2. a global finite-amplitude string or population method to move between basins
   when all local rank-drop coordinates are blocked.

The open problem for this toy is now precise: replace the externally supplied
rank-23 endpoint with a global objective that discovers the same finite
rank-26 -> rank-25 reorganisation autonomously.
