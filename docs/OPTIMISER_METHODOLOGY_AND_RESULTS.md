# Navigating Exact Bilinear-Algorithm Manifolds by Differential Geometry and Basin Search

## Working methodology and results draft

### 1. Overview

We study fast bilinear algorithms as points on the algebraic variety of exact low-rank decompositions of a bilinear tensor. Rather than fixing a target rank and minimizing tensor reconstruction error from a random initialization, the method begins from a known exact algorithm at rank \(R\), moves through the manifold of exact rank-\(R\) algorithms, and searches for boundaries at which one multiplication can be removed. After a successful boundary crossing and exact correction at rank \(R-1\), the procedure repeats.

For matrix multiplication, the target tensor is the usual matrix-multiplication tensor \(T_n\). A rank-\(R\) algorithm is represented as

\[
\widehat T(\theta)=\sum_{r=1}^R a_r\,u_r\otimes v_r\otimes w_r,
\]

with unit-normalized factor directions \(u_r,v_r,w_r\) and a scalar channel amplitude \(a_r\). The exact solution set is

\[
\mathcal M_R=\{\theta:F(\theta)=0\},
\qquad
F(\theta)=\operatorname{vec}(\widehat T(\theta)-T_n).
\]

This representation separates channel scale from direction and makes both amplitude death and channel collision visible.

The final optimiser combines local differential geometry, finite exact-manifold hops, bounded off-manifold basin transitions, and a small Pareto beam of exact basins. Rank reduction is accepted only after the lower-rank system corrects back to essentially machine precision with finite coefficients.

A key methodological distinction is between **development** and **evaluation**. During development, a known rank-23 \(3\times3\) endpoint was used as an investigative oracle to understand failure modes and identify useful global coordinates. The final specialist Pareto controller was then frozen and evaluated endpoint-free on fresh random seeds.

---

## 2. Local geometry of exact algorithms

Let

\[
J(\theta)=DF(\theta)
\]

be the Jacobian of the tensor residual. At an exact solution, the physical tangent space of \(\mathcal M_R\) is approximated by \(\ker J\), after removing the trivial factor-scale gauge through unit-column normalization.

The singular spectrum of \(J\) is used both as a conditioning diagnostic and as a description of local freedom. Small or zero singular values correspond to directions in which an exact algorithm can move without changing the represented tensor to first order.

### 2.1 Channel killability

For each multiplication channel \(r\), define the first-order killability

\[
K_r=\left\|P_{\ker J}\nabla a_r\right\|,
\]

where \(P_{\ker J}\) is the projector onto the tangent space. A useful local death-distance proxy is

\[
D_r=\frac{|a_r|}{K_r}.
\]

Small \(D_r\) means that the channel amplitude can decrease rapidly along an exact first-order motion. Local continuation uses these directions to move amplitudes toward zero while correcting back onto the exact manifold.

This quantity is deliberately treated as a **local** coordinate. In the \(3\times3\) experiments, many trajectories reached an amplitude wall near \(a\approx1\), showing that low local death distance alone does not identify the global basin containing a lower-rank boundary.

### 2.2 Soft effective nullity

Global reorganisation was better correlated with a reduction of excess tangent freedom. Instead of relying on a hard numerical rank threshold, we use a smooth effective nullity

\[
N_\tau(J)
=
n_{\mathrm{param}}
-
\sum_i
\frac{\sigma_i^2}{\sigma_i^2+\tau^2},
\]

where \(\sigma_i\) are singular values of the physical Jacobian and \(\tau\) is inferred from the current spectrum. A candidate basin is evaluated using the parent basin's \(\tau\), so formerly null directions that become weakly constrained contribute continuously rather than jumping discontinuously at an arbitrary rank tolerance.

This quantity is not given a target value. The optimiser only prefers child basins that reduce effective nullity relative to their parent.

---

## 3. Rank-reduction mechanisms

Two distinct rank-reduction mechanisms appeared in the experiments.

### 3.1 Channel collision and fusion

For \(2\times2\) multiplication, naive amplitude death from the schoolbook rank-8 decomposition is locally blocked. The exact path to Strassen instead proceeds through a collision: two rank-one channels become proportional and can then be fused into one channel.

This led to an autonomous collision score based on constrained curvature of pair alignment on the exact-algorithm manifold. Starting from schoolbook \(2\times2\), the geometry selects a symmetry-equivalent opposite-corner pair, follows exact collision ascent, and fuses the pair to recover rank 7.

The same idea scales to the first \(3\times3\) reduction. The schoolbook rank-27 decomposition contains embedded \(2\times2\times2\) cubes. Pair-curvature identifies opposite-corner channel pairs, and following the corresponding exact collision family inside the selected cube gives an exact rank-26 decomposition.

### 3.2 Basin reorganisation followed by amplitude death

Later \(3\times3\) reductions did not occur by simple continuation from the current exact basin. Instead, the optimiser had to reorganise the exact rank-\(R\) representation before deletion became possible.

A useful diagnostic is **deletion susceptibility**. For a candidate channel \(r\), temporarily remove that channel, give the rank-\(R-1\) variables a short bounded relaxation budget, and record the smallest residual reached:

\[
E_r(\theta)
=
\min_{\text{short bounded relaxation}}
\|F_{R-1}\|.
\]

Then

\[
E_{\mathrm{delete}}(\theta)=\min_r E_r(\theta).
\]

This quantity does not itself certify a rank drop. It asks only how close the current basin is to a lower-rank basin after deletion. A rank reduction is accepted only after a separate full lower-rank correction succeeds to strict tolerance.

---

## 4. Global basin navigation

### 4.1 Exact shell hops

When local continuation reaches a wall, the optimiser explores finite tangent motions that are weakly or non-integrable at second order. In development we used an obstruction operator of the form

\[
B(v)=L^\top \dot J[v]N,
\]

where \(N\) and \(L\) span right and left nullspaces of \(J\). Large obstruction directions are tangent at first order but strongly change the Jacobian constraints at second order.

A finite shell step is taken along selected directions, followed by nonlinear correction back to an exact rank-\(R\) solution. These hops can substantially alter tangent nullity and channel mobility while preserving exactness.

### 4.2 Off-manifold tunnels

Exact shell motion alone can remain trapped in one stratum. The optimiser therefore includes a bounded off-manifold tunnel:

\[
\text{exact basin}
\rightarrow
\text{bounded off-manifold move}
\rightarrow
\text{soft shell relaxation}
\rightarrow
\text{return to an exact rank-}R\text{ basin}.
\]

Candidate landings are rejected if coefficients become non-finite or excessively large, or if exact correction fails.

The off-manifold move is a search device, not an accepted approximate algorithm. All retained basins are corrected back to the exact rank-\(R\) manifold before they enter the search frontier.

---

## 5. Pareto beam over exact basins

A single greedy trajectory repeatedly forgot useful basins. The final global policy therefore maintains a small Pareto beam of exact solutions.

Each basin is described by the tuple

\[
\left(
N_\tau(J),
E_{\mathrm{delete}},
D_{\min},
A_{\max}
\right),
\]

where \(D_{\min}=\min_r D_r\) and \(A_{\max}\) bounds coefficient growth. Lower soft nullity, deletion susceptibility and death distance are preferred; amplitude growth is penalised.

The beam width is four. Retention is Pareto-based rather than controlled by one scalar objective, because the useful coordinates are often antagonistic: one basin may have unusually low effective nullity, another unusually low deletion susceptibility, and another unusually low death distance.

### 5.1 Specialist scheduling

Pareto retention alone was insufficient because good specialist basins could survive in the beam without being expanded. The frozen controller therefore explicitly schedules expansions of the current genericity champion, deletion champion, death-distance champion, and periodically a lightly explored or novel basin.

If one basin holds multiple specialist roles, the spare expansion budget is assigned to exploration.

The genericity specialist receives stronger basin-reorganisation moves; the deletion specialist receives short deletion challenges and related local rearrangements; the death specialist receives local continuation toward an amplitude wall followed by a hop.

Retained champions are polished before archiving, and lineage plus expansion counts are recorded. Novelty is used only to avoid obvious revisits, not as a scientific objective.

### 5.2 Acceptance of a rank drop

A channel is never removed merely because its amplitude is small or a short deletion probe looks promising.

A proposed rank-\(R\to R-1\) transition is accepted only when the rank-\(R-1\) parameterisation can be corrected to a stringent tensor residual with finite coefficients. The beam is then reset around the new exact rank-\(R-1\) state and all spectral scales and local diagnostics are recomputed from scratch.

---

## 6. Development trajectory

For \(2\times2\), schoolbook rank 8 was found to be strongly stable against naive amplitude death. The physical exact tangent space has no first-order amplitude-decrease direction at the schoolbook point, and the second-order amplitude curvature is positive semidefinite. A string calculation between schoolbook and Strassen then exposed a collision-and-fusion mechanism, from which an exact one-parameter schoolbook-to-Strassen homotopy and an autonomous collision search were obtained.

For \(3\times3\), collision geometry autonomously gave the first reduction \(27\to26\). Direct local continuation at rank 26 repeatedly hit amplitude walls. Guided experiments using a known rank-23 endpoint were then used as a diagnostic oracle to identify two global features of successful trajectories: reduction of excess tangent freedom and increasing susceptibility to deletion. The endpoint was subsequently removed from the optimiser.

Blind off-manifold basin transitions produced \(26\to25\), and the hybrid state machine produced \(25\to24\). Rank 24 exposed repeated basin cycling. Adding deletion susceptibility alone did not solve the problem; adding soft-nullity genericisation helped but remained greedy. A Pareto beam preserved incompatible kinds of progress, and specialist scheduling finally produced a reliable \(24\to23\) transition.

The final specialist Pareto policy was frozen before replication experiments.

---

## 7. Frozen-policy replication results

Five fresh seeds were selected before observing their outcomes:

\[
101,\quad211,\quad307,\quad401,\quad503.
\]

Each replicate independently began with the schoolbook rank-27 decomposition, performed an autonomous symmetry-equivalent collision reduction to rank 26, and then ran the same frozen specialist Pareto controller with only the RNG seed changed.

All five runs reached rank 23:

| seed | path | beam generations | final residual | max \(|a|\) |
|---:|:---|---:|---:|---:|
| 101 | \(26\to25\to24\to23\) | 12 | \(2.38\times10^{-13}\) | 5.138 |
| 211 | \(26\to25\to24\to23\) | 12 | \(3.61\times10^{-15}\) | 4.608 |
| 307 | \(26\to25\to24\to23\) | 25 | \(3.11\times10^{-10}\) | 3.856 |
| 401 | \(26\to25\to24\to23\) | 12 | \(8.31\times10^{-10}\) | 3.572 |
| 503 | \(26\to25\to24\to23\) | 9 | \(3.23\times10^{-15}\) | 3.593 |

The generation counts vary from 9 to 25, suggesting that the success is not a fixed scripted sequence despite the common rank path.

All endpoints can be polished back to residuals of order \(10^{-15}\), including the two endpoints that terminated the beam run at approximately \(10^{-10}\).

---

## 8. Exactification

A numerically exact rank-23 decomposition is not treated as a final certificate.

The exactification pipeline first searches the full \(GL(3)^3\) matrix-multiplication isotropy group for a well-conditioned incidence-based gauge. Repeated projective row and column directions among rank-one factors provide a finite basis-selection problem. For the first discovered endpoint, this produced transforms with condition numbers close to one and converted 405 of 644 gauge-fixed scalar coordinates into structural zeros.

After locking the structural zero pattern and per-channel gauge pivots, the Brent equations are reduced to a smaller nonlinear system. The reduced system generally has nonzero local family dimension. Small rational locks are introduced along those family freedoms to select a nearby arithmetic representative, and the remaining coordinates are refined at high precision.

Arithmetic recognition then searches for a common algebraic number field. Exact field arithmetic in

\[
\mathbb Q[\alpha]/(p(\alpha))
\]

is used to verify all 729 Brent identities. The exact representative need not be the identical floating-point point produced by the optimiser; it is a nearby point on the same local exact solution family. Family-move norms are therefore reported explicitly.

The first exactified endpoint lies in

\[
\mathbb Q\!\left(\sqrt{85\,213\,608\,769}\right),
\]

with 594 of 644 gauge-fixed coefficients rational and 50 genuinely quadratic. All 729 identities vanish exactly.

Among the five frozen-seed endpoints, exact certificates have so far been obtained for seeds 101, 211 and 401. Seed 101 lies in a cubic field and seeds 211 and 401 admit rational representatives. Seeds 307 and 503 remain under arithmetic-field analysis; this is an exactification limitation rather than evidence against their numerical rank-23 status.

---

## 9. Equivalence classification against known schemes

Exact factor-matrix ranks are preserved under channel permutation, per-channel CP scaling, the \(GL(3)^3\) matrix-multiplication isotropy action, and tensor-leg permutation. This gives a cheap exact orbit invariant: the canonical 23-channel pattern

\[
\left\{
(\operatorname{rank}A_r,\operatorname{rank}B_r,\operatorname{rank}C_r)
\right\}_{r=1}^{23}.
\]

A tar archive containing 17,376 JKU rank-23 \(3\times3\) schemes was parsed directly. All 17,376 files were successfully parsed with exact integer rank arithmetic.

For the original exact certificate, the archive funnel was

\[
17376
\rightarrow165
\rightarrow30
\rightarrow0
\]

through progressively stronger factor-rank invariants, with zero schemes sharing the full canonical factor-rank pattern.

The exact representatives for seeds 101 and 211 likewise have zero full-pattern matches in the archive.

Seed 401 has seven archive schemes with the same canonical full factor-rank pattern. These seven were tested using a stronger directed sandwich-incidence invariant, colouring each ordered channel pair by

\[
\left(
\operatorname{rank}(A_rB_s),
\operatorname{rank}(B_rC_s),
\operatorname{rank}(C_rA_s)
\right).
\]

All seven fail even the exact edge-label histogram, and therefore all are rigorously inequivalent to seed 401.

At present, the original endpoint and exactified seeds 101, 211 and 401 are four mutually structurally distinct exact rank-23 algorithms whose equivalence classes are absent from the 17,376-entry JKU archive.

This archive comparison does not establish that the corresponding continuous solution families are previously unknown, nor does it cover every rank-23 scheme published after the archive was assembled.

---

## 10. Post-discovery arithmetic-complexity navigation

Rank and arithmetic circuit complexity are different objectives. Once rank 23 had been reached and exactified, a separate fixed-rank campaign asked whether the same local exact-family geometry could be used to find sparser representatives.

The first pass optimises support-derived addition count: a linear form with support \(k\) contributes \(k-1\) additions/subtractions. Tensor equality remains a hard constraint. The search projects a smooth sparsity objective into the tangent space, corrects back to the exact rank-23 manifold, preserves existing structural zeros, and attempts to snap newly tiny coefficients to exact zero.

Starting from the exact seed-211 and seed-401 representatives, the first campaign obtained:

| seed | starting exact naive additions | numerical sparse winner | exact sparse representative | greedy exact CSE |
|---:|---:|---:|---:|:---|
| 211 | 143 | 127 | **124** | 94 -> **82** |
| 401 | 137 | 128 | **109** | 87 -> **71** |

Seed 211's sparse representative is rational. Seed 401 exactifies in \(\mathbb Q(\sqrt{23})\), equivalently using a generator \(\alpha\) satisfying \(23\alpha^2-4=0\). All 729 Brent identities verify exactly, and every zero deliberately frozen by the numerical complexity campaign remains an exact symbolic zero.

The additional drop from 128 numerical additions to 109 exact additions for seed 401 is noteworthy: exactification is not merely certifying a thresholded numerical point, but can move along the same constrained exact family to a considerably simpler arithmetic representative.

The CSE numbers are deterministic exact straight-line-program upper bounds, not optimal additive-complexity proofs. They also do not charge arbitrary scalar constant multiplications, so they should not be compared directly with ternary low-addition schemes on total scalar-operation cost.

---

## 11. Complexity guidance during rank discovery

A matched ablation tested whether arithmetic simplicity should be included during the \(26\to23\) discovery process itself. The local operators, shared rank-26 starting checkpoint, beam width, and expansion budget were held fixed. Complexity only weakly influenced beam retention/expansion; no structural zeros were snapped or frozen above rank 23.

Baseline, constant-weak, and rank-adaptive policies all reached rank 23 on all five frozen seeds and all had median structural endpoint complexity 566 additions. Four seeds reproduced the baseline generation count under the guided policies. Seed 211 was the exception: the baseline reached rank 23 in 12 beam generations, constant weak guidance required 29, and produced no simpler endpoint. A delayed policy that activates only at rank 24 returned seed 211 to the 12-generation baseline path; delayed runs on seeds 307 and 503 also matched baseline.

At the tested strength, simplicity therefore provides no demonstrated discovery benefit and can perturb a useful earlier basin trajectory. This contrasts with the strong post-discovery simplification result and supports a staged architecture: **discover rank first, then optimise arithmetic complexity on the resulting exact family**.

---

## 12. Current interpretation

The main result is not an improved matrix-multiplication exponent or a new multiplication-count record: rank 23 for \(3\times3\) matrix multiplication has long been known.

The interesting result is algorithmic. A frozen search procedure beginning from the schoolbook algorithm repeatedly navigates exact-algorithm manifolds, crosses basin boundaries, and locates lower-rank boundaries without being supplied a target rank-23 decomposition. Five fresh-seed runs all reached rank 23, and exactification shows that the method is not merely returning one familiar equivalence class.

A concise current claim is:

> Starting from the schoolbook rank-27 decomposition of the \(3\times3\) matrix-multiplication tensor, a differential-geometric and Pareto basin-search procedure reproducibly discovers rank-23 decompositions. In five frozen-policy fresh-seed trials, all five reached rank 23. Multiple resulting local solution families contain exact rank-23 algorithms whose isotropy classes are absent from a 17,376-scheme reference archive.

The strongest unresolved questions are whether these exact representatives belong to previously documented continuous families, how broadly the method transfers to other bilinear tensors, whether a comparable result can be obtained by much simpler optimisation, and whether the same machinery can locate a rank-22 boundary.

The new complexity experiments sharpen the methodological picture. Arithmetic simplicity was not useful as a weak discovery preference in the frozen five-seed ablation, but became highly effective once the search was restricted to an already-discovered rank-23 family. This makes the search look less like one monolithic objective and more like a sequence of geometric tasks with different useful coordinates.

---

## 13. Immediate next experiments

The most important control is now a deliberately simple optimisation baseline. Starting from the same schoolbook decomposition, plain gradient descent or Adam should be given matched seeds and an approximate matched compute budget, with a second baseline adding an elementary sparsity/death penalty. If these methods remain trapped at higher rank while the basin controller repeatedly reaches rank 23, the role of the geometric/global machinery becomes much clearer.

A separate frozen rank-22 campaign has been explored from exact rank-23 starting points without an accepted crossing so far. That is only negative search evidence, not a lower bound. Any future \(23\to22\) success must require finite coefficients, high-precision refinement, no border-rank blow-up/cancellation, and exact verification of all 729 Brent identities.

For arithmetic complexity, the next step is no longer to establish that simplification is possible. It is to improve the cost model: coefficient alphabet, nontrivial scalar-multiplication cost, stronger CSE/straight-line-program optimisation, and repeated simplify--exactify cycles.

A complementary scaling experiment is \(4\times4\) multiplication. The first target should be a known rank reduction as a sanity check for the geometry and sparse-linear-algebra implementation at larger scale. Only after reproducing a known boundary should the method be used to explore a previously unknown one.
