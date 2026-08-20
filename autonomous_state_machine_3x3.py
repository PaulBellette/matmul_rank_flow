"""Exploratory endpoint-free rank-reduction controller for 3x3 matmul.

This is intentionally a hybrid dynamical algorithm rather than a single scalar
optimizer.  It alternates four qualitatively different operations:

    CONTINUE_TO_WALL
        Follow the exact algorithm manifold while decreasing a locally movable
        channel amplitude.

    HOP
        Use a tangent direction with large second-order obstruction, then solve
        for a *different* exact solution on a finite-radius shell.

    OFF_MANIFOLD_HOP
        If exact shell landings are exhausted, keep a soft finite-radius shell
        while allowing a bounded tensor residual.  Anneal back toward exactness
        and only accept a genuinely different exact rank-R basin.

    DELETE_PROBE
        Temporarily leave the exact manifold and clamp one channel through the
        amplitude wall.  Residual relaxation is allowed, but coefficient blowup
        is guarded against.

    DROP
        If the off-manifold probe lands near a true lower-rank basin, remove
        the clamped channel and project directly onto the exact rank-(R-1)
        manifold.

No rank-23 target or known target channel is used by this controller.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Iterable

import torch

from curvature_flow import (
    constraint_jacobian,
    constraint_second_directional,
    correct_constraints,
    physical_constraints,
    physical_killability,
    tangent_basis,
)
from geometry_flow import amp_index, pack, residual_vector, robust_svd, unpack
from rankflow import mm_tensor


torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent
DEFAULT_START = ROOT / "reference_guided_cascade" / "rank26.pt"


class Phase(str, Enum):
    CONTINUE_TO_WALL = "continue_to_wall"
    HOP = "hop"
    OFF_MANIFOLD_HOP = "off_manifold_hop"
    DELETE_PROBE = "delete_probe"
    DROP = "drop"
    DONE = "done"
    FAILED = "failed"


@dataclass
class ControllerConfig:
    n: int = 3
    goal_rank: int = 23
    rcond: float = 1.0e-10
    exact_tol: float = 1.0e-10
    coefficient_cap: float = 8.0

    # Exact amplitude continuation.
    amp_step: float = 0.05
    min_amp_step: float = 0.0025
    wall_killability: float = 2.5e-2
    wall_amplitude: float = 1.03
    continuation_max_steps: int = 40

    # Finite exact basin hop.
    hop_trials: int = 5
    hop_radius: float = 2.5
    hop_radius_growth: float = 1.25
    hop_max_radius: float = 8.0
    hop_obstruction_trials: int = 6

    # Off-manifold basin tunnel used when exact shell hops are exhausted.
    # The tunnel keeps a soft shell so relaxation cannot simply fall back to
    # the current exact branch, then asks the exact corrector to finish.
    offhop_trials: int = 3
    offhop_lr: float = 1.5e-2
    offhop_steps_per_stage: int = 350
    offhop_shell_weights: tuple[float, ...] = (8.0, 3.0, 1.0, 0.3)
    offhop_min_distance_fraction: float = 0.35
    offhop_exact_max_iters: int = 60

    # Cheap deletion-susceptibility score for ranking tunnel landings.
    # This is deliberately much cheaper than DELETE_PROBE: clamp a few
    # promising channels directly to zero, relax briefly, and ask how much
    # tensor residual remains.  It may rank basins but may never declare a drop.
    susceptibility_channels: int = 3
    susceptibility_steps: int = 140
    susceptibility_lr: float = 2.0e-2
    susceptibility_noise: float = 5.0e-4
    susceptibility_weight_death_distance: float = 1.0e-2
    archive_novelty_tol: float = 2.0e-3

    # Generic basin objective.  At any rank R, use the current exact state to
    # define a spectral gap between physical Jacobian null modes and positive
    # modes.  Tunnel/exact-hop candidates are rewarded for opening those soft
    # modes (lower effective nullity).  Deletion susceptibility becomes
    # competitive only when it is genuinely small.  Nothing here is tied to a
    # particular target nullity or to R=24.
    genericity_weight: float = 6.0
    genericity_deletion_weight: float = 1.0
    genericity_deletion_scale: float = 1.0e-1
    genericity_death_distance_weight: float = 2.0e-2
    genericity_coefficient_weight: float = 5.0e-2

    # Pareto beam over exact basins.  The beam is deliberately tiny: the goal
    # is to stop forgetting useful branches, not to turn the experiment into a
    # brute-force population search.
    beam_width: int = 4
    # Expand metric specialists explicitly rather than choosing parents by a
    # single balanced rank.  Three specialists are cheap enough to expand every
    # generation; every few generations an additional lightly-explored basin is
    # admitted when beam capacity allows.
    beam_expand: int = 3
    beam_explore_every: int = 3
    beam_exact_children: int = 3
    beam_genericity_offhop_children: int = 2
    beam_delete_probe_states: int = 1
    beam_delete_every: int = 2
    beam_delete_trigger: float = 0.35
    beam_min_soft_gain: float = 0.25
    beam_offhop_children: int = 1
    beam_polish_tol: float = 1.0e-13
    beam_polish_max_iters: int = 8

    # Off-manifold deletion probe.
    delete_every_hops: int = 2
    delete_lr: float = 2.0e-2
    delete_steps_per_stage: int = 700
    delete_final_steps: int = 2500
    delete_accept_residual: float = 1.0e-2
    delete_exact_tol: float = 1.0e-9
    clamp_fractions: tuple[float, ...] = (0.8, 0.6, 0.4, 0.2, 0.0)

    max_cycles: int = 16
    seed: int = 0


@dataclass
class Analysis:
    rank: int
    residual: float
    jacobian_rank: int
    tangent_dim: int
    sigma_min_positive: float
    condition_positive: float
    min_abs_amplitude: float
    max_abs_amplitude: float
    best_channel: int
    best_amplitude: float
    best_killability: float
    best_death_distance: float


@dataclass
class Transition:
    cycle: int
    phase: str
    rank: int
    channel: int
    residual_before: float
    residual_after: float
    amplitude_before: float
    amplitude_after: float
    killability_before: float
    killability_after: float
    max_amplitude_after: float
    note: str


@dataclass
class GenericitySpectrum:
    tau: float
    soft_nullity: float
    hard_nullity: int
    sigma_min_positive: float
    sigma_max_null: float


@dataclass
class BasinState:
    basin_id: int
    theta: torch.Tensor
    rank: int
    parent_id: int
    generation: int
    source: str
    analysis: Analysis
    soft_nullity: float
    susceptibility: float
    susceptibility_channel: int
    max_abs_amplitude: float
    fingerprint: torch.Tensor
    expansion_count: int = 0
    last_expanded_generation: int = -1

    @property
    def death_distance(self) -> float:
        return self.analysis.best_death_distance


@dataclass
class BeamEvent:
    generation: int
    rank: int
    basin_id: int
    parent_id: int
    source: str
    soft_nullity: float
    hard_nullity: int
    susceptibility: float
    susceptibility_channel: int
    death_distance: float
    max_abs_amplitude: float
    residual: float
    accepted: bool
    note: str


def genericity_spectrum(
    theta: torch.Tensor,
    rank: int,
    cfg: ControllerConfig,
    *,
    tau: float | None = None,
) -> GenericitySpectrum:
    """Smooth effective nullity of the physical constraint Jacobian.

    When ``tau`` is omitted, infer it from the current state's own spectral
    gap: the geometric mean of the smallest positive and largest numerical-null
    singular values under the controller's existing ``rcond``.  Candidate
    basins are then evaluated at the *home state's same tau*, so opening a null
    mode continuously lowers ``soft_nullity``.

    This is rank-generic: no desired nullity, decomposition, or target rank is
    encoded in the score.
    """
    target = mm_tensor(cfg.n, theta.device)
    J = constraint_jacobian(theta, target, cfg.n, rank)
    _, S, _ = robust_svd(J, full_matrices=False)
    cols = J.shape[1]
    if len(S) == 0:
        return GenericitySpectrum(1.0, float(cols), cols, 0.0, 0.0)

    s0 = float(S[0])
    cutoff = cfg.rcond * max(s0, 1.0e-30)
    positive = S[S > cutoff]
    nullish = S[S <= cutoff]
    hard_nullity = cols - int(positive.numel())
    sigma_min_positive = float(positive[-1]) if len(positive) else 0.0
    sigma_max_null = float(nullish[0]) if len(nullish) else 0.0

    if tau is None:
        if sigma_min_positive > 0.0 and sigma_max_null > 0.0:
            tau = math.sqrt(sigma_min_positive * sigma_max_null)
        elif sigma_min_positive > 0.0:
            # No resolved null cluster.  Keep the smoothing scale below the
            # weakest positive mode while remaining above floating-point dust.
            tau = max(cutoff, sigma_min_positive * 1.0e-3)
        else:
            tau = max(cutoff, torch.finfo(theta.dtype).eps * max(s0, 1.0))
    tau = max(float(tau), torch.finfo(theta.dtype).tiny)

    # Includes structural right-nullity automatically when cols > len(S).
    resolved_rank = float(((S * S) / (S * S + tau * tau)).sum())
    soft_nullity = float(cols - resolved_rank)
    return GenericitySpectrum(
        tau=tau,
        soft_nullity=soft_nullity,
        hard_nullity=hard_nullity,
        sigma_min_positive=sigma_min_positive,
        sigma_max_null=sigma_max_null,
    )


def generic_basin_score(
    *,
    candidate_soft_nullity: float,
    home_soft_nullity: float,
    susceptibility: float | None,
    death_distance: float,
    max_abs_amplitude: float,
    cfg: ControllerConfig,
) -> float:
    """Lower-is-better rank-generic basin score.

    Genericisation dominates while deletion is hard.  Once a cheap deletion
    probe falls by orders of magnitude, the logarithmic susceptibility term
    naturally takes over without a hand-coded phase or target nullity.
    """
    soft_ratio = candidate_soft_nullity / max(home_soft_nullity, 1.0)
    score = cfg.genericity_weight * soft_ratio
    if susceptibility is not None and math.isfinite(susceptibility):
        score += cfg.genericity_deletion_weight * math.log10(
            1.0 + max(susceptibility, 0.0) / cfg.genericity_deletion_scale
        )
    score += cfg.genericity_death_distance_weight * math.log1p(max(death_distance, 0.0))
    score += cfg.genericity_coefficient_weight * (max_abs_amplitude / max(cfg.coefficient_cap, 1.0e-12))
    return score



def _finite_metric(x: float) -> float:
    return float(x) if math.isfinite(float(x)) else 1.0e30


def basin_metrics_tuple(state: BasinState) -> tuple[float, float, float, float]:
    """Lower-is-better Pareto tuple for exact basins."""
    return (
        _finite_metric(state.soft_nullity),
        _finite_metric(state.susceptibility),
        _finite_metric(state.death_distance),
        _finite_metric(state.max_abs_amplitude),
    )


def pareto_dominates(a: BasinState, b: BasinState, *, eps: float = 1.0e-9) -> bool:
    """True when ``a`` is no worse on every basin metric and better on one."""
    ma = basin_metrics_tuple(a)
    mb = basin_metrics_tuple(b)
    no_worse = all(x <= y + eps for x, y in zip(ma, mb))
    strictly = any(x < y - eps for x, y in zip(ma, mb))
    return no_worse and strictly


def pareto_front(states: list[BasinState]) -> list[BasinState]:
    out: list[BasinState] = []
    for i, s in enumerate(states):
        if any(i != j and pareto_dominates(t, s) for j, t in enumerate(states)):
            continue
        out.append(s)
    return out


def _rank_positions(states: list[BasinState], key) -> dict[int, int]:
    ordered = sorted(states, key=lambda s: (_finite_metric(key(s)), s.basin_id))
    return {s.basin_id: i for i, s in enumerate(ordered)}


def beam_priority_order(states: list[BasinState]) -> list[BasinState]:
    """Balanced order that preserves specialists rather than one scalar score."""
    if not states:
        return []
    ranks = [
        _rank_positions(states, lambda s: s.soft_nullity),
        _rank_positions(states, lambda s: s.susceptibility),
        _rank_positions(states, lambda s: s.death_distance),
        _rank_positions(states, lambda s: s.max_abs_amplitude),
    ]
    return sorted(
        states,
        key=lambda s: (
            sum(r[s.basin_id] for r in ranks),
            min(r[s.basin_id] for r in ranks),
            s.basin_id,
        ),
    )


def specialist_expansion_schedule(
    states: list[BasinState],
    generation: int,
    cfg: ControllerConfig,
) -> list[tuple[BasinState, tuple[str, ...]]]:
    """Choose beam parents by *why* each basin is valuable.

    Pareto retention prevents us from forgetting specialists; this schedule
    prevents those specialists from starving.  The genericity, deletion and
    death-distance champions are always requested.  If one basin holds several
    titles, the freed slot is filled by the least-expanded/newest basin.  Every
    ``beam_explore_every`` generations we allow one extra exploration slot (up
    to the beam width).
    """
    if not states:
        return []

    by_id: dict[int, tuple[BasinState, set[str]]] = {}

    def add(state: BasinState, role: str):
        if state.basin_id not in by_id:
            by_id[state.basin_id] = (state, set())
        by_id[state.basin_id][1].add(role)

    add(min(states, key=lambda s: (_finite_metric(s.soft_nullity), s.basin_id)), "genericity")
    add(min(states, key=lambda s: (_finite_metric(s.susceptibility), s.basin_id)), "deletion")
    add(min(states, key=lambda s: (_finite_metric(s.death_distance), s.basin_id)), "death")

    base_limit = max(1, cfg.beam_expand)
    explore_due = cfg.beam_explore_every > 0 and generation % cfg.beam_explore_every == 0
    limit = min(cfg.beam_width, base_limit + (1 if explore_due else 0))

    # Fill specialist collisions/spare capacity with states that have received
    # the least compute.  For equal counts prefer newer states so fresh branches
    # get at least one chance before old basins monopolise the beam.
    for state in sorted(
        states,
        key=lambda s: (s.expansion_count, -s.generation, s.last_expanded_generation, s.basin_id),
    ):
        if len(by_id) >= limit:
            break
        if state.basin_id not in by_id:
            add(state, "explore")

    ordered_roles = {"genericity": 0, "deletion": 1, "death": 2, "explore": 3}
    scheduled = list(by_id.values())
    scheduled.sort(
        key=lambda item: (
            min(ordered_roles[r] for r in item[1]),
            item[0].expansion_count,
            item[0].basin_id,
        )
    )
    return [(state, tuple(sorted(roles, key=lambda r: ordered_roles[r]))) for state, roles in scheduled]


def pareto_beam_prune(states: list[BasinState], width: int) -> list[BasinState]:
    """Pareto-prune while explicitly retaining metric specialists.

    A pure scalar score caused the greedy controller to forget low-nullity
    basins.  Here we first keep the non-dominated set, then guarantee slots for
    the best state under each individual metric before filling by balanced
    rank-sum.  This is intentionally tiny and deterministic.
    """
    if width <= 0 or not states:
        return []
    front = pareto_front(states)
    pool = front if front else states
    chosen: list[BasinState] = []
    seen: set[int] = set()

    specialists = [
        min(pool, key=lambda s: (_finite_metric(s.soft_nullity), s.basin_id)),
        min(pool, key=lambda s: (_finite_metric(s.susceptibility), s.basin_id)),
        min(pool, key=lambda s: (_finite_metric(s.death_distance), s.basin_id)),
        min(pool, key=lambda s: (_finite_metric(s.max_abs_amplitude), s.basin_id)),
    ]
    for state in specialists:
        if state.basin_id not in seen and len(chosen) < width:
            chosen.append(state)
            seen.add(state.basin_id)

    for state in beam_priority_order(pool):
        if state.basin_id not in seen and len(chosen) < width:
            chosen.append(state)
            seen.add(state.basin_id)

    # If the non-dominated front was smaller than the beam, admit the strongest
    # dominated alternatives rather than wasting capacity.
    if len(chosen) < width:
        for state in beam_priority_order(states):
            if state.basin_id not in seen and len(chosen) < width:
                chosen.append(state)
                seen.add(state.basin_id)
    return chosen


def fingerprint_is_novel(
    fp: torch.Tensor,
    archive: list[torch.Tensor],
    tol: float,
) -> tuple[bool, float]:
    d = min(
        (fingerprint_distance(fp, old) for old in archive if old.numel() == fp.numel()),
        default=math.inf,
    )
    return d >= tol, d


def make_basin_state(
    theta: torch.Tensor,
    rank: int,
    cfg: ControllerConfig,
    *,
    basin_id: int,
    parent_id: int,
    generation: int,
    source: str,
    tau: float,
    seed: int,
    probe_susceptibility: bool = True,
) -> BasinState:
    A = analyse(theta, rank, cfg)
    G = genericity_spectrum(theta, rank, cfg, tau=tau)
    if probe_susceptibility:
        susceptibility, channel, _ = deletion_susceptibility(theta, rank, cfg, seed=seed)
    else:
        susceptibility, channel = math.inf, A.best_channel
    return BasinState(
        basin_id=basin_id,
        theta=theta.detach().clone(),
        rank=rank,
        parent_id=parent_id,
        generation=generation,
        source=source,
        analysis=A,
        soft_nullity=G.soft_nullity,
        susceptibility=susceptibility,
        susceptibility_channel=channel,
        max_abs_amplitude=A.max_abs_amplitude,
        fingerprint=basin_fingerprint(theta, rank, cfg),
        expansion_count=0,
        last_expanded_generation=-1,
    )


def basin_state_row(state: BasinState, *, accepted: bool = True, note: str = "") -> dict:
    return {
        "generation": state.generation,
        "rank": state.rank,
        "basin_id": state.basin_id,
        "parent_id": state.parent_id,
        "source": state.source,
        "soft_nullity": state.soft_nullity,
        "hard_nullity": state.analysis.tangent_dim,
        "susceptibility": state.susceptibility,
        "susceptibility_channel": state.susceptibility_channel,
        "death_distance": state.death_distance,
        "best_channel": state.analysis.best_channel,
        "best_amplitude": state.analysis.best_amplitude,
        "best_killability": state.analysis.best_killability,
        "max_abs_amplitude": state.max_abs_amplitude,
        "residual": state.analysis.residual,
        "expansion_count": state.expansion_count,
        "last_expanded_generation": state.last_expanded_generation,
        "accepted": accepted,
        "note": note,
    }


def load_theta(path: Path) -> tuple[torch.Tensor, int]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        q = obj.detach().clone().to(torch.float64)
        # infer R from p = 3*n^2*R + R = 28R for n=3
        if q.numel() % 28 != 0:
            raise ValueError(f"cannot infer rank from tensor length {q.numel()}")
        return q, q.numel() // 28
    if "theta" in obj:
        q = obj["theta"].detach().clone().to(torch.float64)
        if q.numel() % 28 != 0:
            raise ValueError(f"cannot infer rank from tensor length {q.numel()}")
        return q, q.numel() // 28
    if all(k in obj for k in ("U", "V", "W", "a")):
        q = pack(obj["U"], obj["V"], obj["W"], obj["a"])
        return q.to(torch.float64), int(obj["a"].numel())
    raise ValueError(f"unsupported checkpoint format: {path}")


def save_theta(path: Path, theta: torch.Tensor, rank: int, **meta):
    U, V, W, a = unpack(theta.detach().cpu(), 3, rank)
    payload = {"theta": theta.detach().cpu(), "U": U, "V": V, "W": W, "a": a, "rank": rank}
    payload.update(meta)
    torch.save(payload, path)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Hybrid phases can add diagnostics only after a landing has been found.
    # Preserve the union of fields rather than assuming row 0 is the schema.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyse(theta: torch.Tensor, rank: int, cfg: ControllerConfig) -> Analysis:
    target = mm_tensor(cfg.n, theta.device)
    kills, info = physical_killability(theta, target, cfg.n, rank, rcond=cfg.rcond)
    _, _, _, a = unpack(theta, cfg.n, rank)

    # Prefer a small amplitude that still has a usable exact tangent direction.
    # This is a local estimate of parameter distance required to reach a=0.
    scored = []
    for row in kills:
        k = float(row["killability"])
        amp = abs(float(row["amplitude"]))
        d = amp / max(k, 1.0e-12)
        scored.append((d, -k, row))
    scored.sort(key=lambda x: (x[0], x[1]))
    best = scored[0][2]

    return Analysis(
        rank=rank,
        residual=float(residual_vector(theta, target, cfg.n, rank).norm()),
        jacobian_rank=info.rank,
        tangent_dim=info.nullity,
        sigma_min_positive=info.sigma_min_positive,
        condition_positive=info.condition_positive,
        min_abs_amplitude=float(a.abs().min()),
        max_abs_amplitude=float(a.abs().max()),
        best_channel=int(best["channel"]),
        best_amplitude=float(best["amplitude"]),
        best_killability=float(best["killability"]),
        best_death_distance=abs(float(best["amplitude"])) / max(float(best["killability"]), 1.0e-12),
    )


def corrected_with_fixed_amplitude(
    theta: torch.Tensor,
    rank: int,
    channel: int,
    target_amplitude: float,
    cfg: ControllerConfig,
) -> tuple[torch.Tensor, bool, float]:
    target = mm_tensor(cfg.n, theta.device)
    idx = amp_index(cfg.n, rank, channel)
    x = theta.detach().clone()
    x[idx] = target_amplitude

    def constraints(q: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [physical_constraints(q, target, cfg.n, rank), (q[idx] - target_amplitude).reshape(1)]
        )

    total = 0.0
    for _ in range(22):
        r = constraints(x).detach()
        rn = float(r.norm())
        if rn <= cfg.exact_tol:
            return x, True, total
        z = x.detach().clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(constraints, z, vectorize=True).detach()
        # Truncated minimum-norm solve.
        U, S, Vh = robust_svd(J, full_matrices=False)
        cutoff = cfg.rcond * float(S[0]) if len(S) else cfg.rcond
        keep = S > cutoff
        if not bool(keep.any()):
            break
        delta = Vh[keep, :].T @ (((U[:, keep].T @ (-r)) / S[keep]))
        dn = float(delta.norm())
        if not math.isfinite(dn) or dn == 0.0:
            break
        accepted = False
        alpha = 1.0
        for _ in range(12):
            cand = x + alpha * delta
            cand[idx] = target_amplitude
            if float(constraints(cand).norm()) < rn:
                x = cand
                total += alpha * dn
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    return x, float(constraints(x).norm()) <= cfg.exact_tol, total


def continue_to_wall(
    theta: torch.Tensor,
    rank: int,
    channel: int,
    cfg: ControllerConfig,
) -> tuple[torch.Tensor, list[dict], str]:
    target = mm_tensor(cfg.n, theta.device)
    rows: list[dict] = []
    x = theta.detach().clone()
    step = cfg.amp_step

    for iteration in range(cfg.continuation_max_steps):
        A = analyse(x, rank, cfg)
        _, _, _, a = unpack(x, cfg.n, rank)
        amp = float(a[channel])
        kills, _ = physical_killability(x, target, cfg.n, rank, rcond=cfg.rcond)
        k = float(kills[channel]["killability"])
        rows.append(
            {
                "iteration": iteration,
                "channel": channel,
                "amplitude": amp,
                "killability": k,
                "residual": A.residual,
                "jacobian_rank": A.jacobian_rank,
                "tangent_dim": A.tangent_dim,
                "sigma_min_positive": A.sigma_min_positive,
                "max_abs_amplitude": A.max_abs_amplitude,
                "amp_step": step,
            }
        )
        if abs(amp) <= 1.0e-7:
            return x, rows, "zero_reached"
        if k <= cfg.wall_killability:
            return x, rows, "killability_wall"
        if abs(amp) <= cfg.wall_amplitude:
            return x, rows, "amplitude_wall"

        sign = 1.0 if amp >= 0 else -1.0
        wanted = amp - sign * min(step, max(0.0, abs(amp) - cfg.wall_amplitude))
        if abs(wanted - amp) < cfg.min_amp_step * 0.25:
            return x, rows, "amplitude_wall"

        trial_step = abs(wanted - amp)
        accepted = False
        while trial_step >= cfg.min_amp_step:
            target_amp = amp - sign * trial_step
            cand, ok, _ = corrected_with_fixed_amplitude(x, rank, channel, target_amp, cfg)
            if ok:
                try:
                    ca = analyse(cand, rank, cfg)
                except (torch._C._LinAlgError, RuntimeError):
                    # A corrected candidate may sit on an extremely clustered
                    # Jacobian spectrum.  If even robust_svd cannot analyse it,
                    # treat that as a continuation wall and try a smaller step
                    # rather than killing the entire exploratory run.
                    ca = None
                if ca is not None and ca.max_abs_amplitude <= cfg.coefficient_cap:
                    x = cand
                    step = min(cfg.amp_step, trial_step * 1.3)
                    accepted = True
                    break
            trial_step *= 0.5
        if not accepted:
            return x, rows, "continuation_stalled"

    return x, rows, "max_steps"


def obstruction_direction(
    theta: torch.Tensor,
    rank: int,
    cfg: ControllerConfig,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float, int]:
    """Pick a tangent direction with large second-order non-integrability.

    For q in ker J, h=D^2G[q,q].  Only the component of h outside range(J)
    cannot be cancelled by a normal acceleration.  Its norm is a direct local
    measure of how strongly the linearized exact manifold is lying to us.
    """
    target = mm_tensor(cfg.n, theta.device)
    J = constraint_jacobian(theta, target, cfg.n, rank)
    U, S, Vh = robust_svd(J, full_matrices=True)
    cutoff = cfg.rcond * float(S[0]) if len(S) else cfg.rcond
    r = int((S > cutoff).sum())
    N = Vh[r:, :].T
    L = U[:, r:]
    if N.shape[1] == 0:
        raise RuntimeError("no exact tangent directions")

    best_q = None
    best_score = -1.0
    best_trial = -1
    for trial in range(cfg.hop_obstruction_trials):
        coeff = torch.randn(N.shape[1], generator=generator, dtype=theta.dtype, device=theta.device)
        q = N @ coeff
        q /= q.norm() + 1.0e-30
        h = constraint_second_directional(theta, q, target, cfg.n, rank)
        score = float((L.T @ h).norm()) if L.shape[1] else 0.0
        if score > best_score:
            best_score = score
            best_q = q
            best_trial = trial
    assert best_q is not None
    return best_q, best_score, best_trial


def finite_shell_hop(
    theta: torch.Tensor,
    rank: int,
    cfg: ControllerConfig,
    *,
    radius: float,
    seed: int,
) -> tuple[torch.Tensor | None, list[dict]]:
    target = mm_tensor(cfg.n, theta.device)
    gen = torch.Generator(device=theta.device)
    gen.manual_seed(seed)
    rows: list[dict] = []
    best_theta = None
    best_score = math.inf
    home_generic = genericity_spectrum(theta, rank, cfg)

    for trial in range(cfg.hop_trials):
        q, obstruction, _ = obstruction_direction(theta, rank, cfg, generator=gen)
        # Try both sides of the obstructed tangent direction.  The shell
        # constraint prevents the corrector from simply falling back home.
        for sign in (-1.0, 1.0):
            pred = theta + sign * radius * q
            cand, ok, iters, rn, correction = correct_constraints(
                pred,
                target,
                cfg.n,
                rank,
                center=theta,
                radius=radius,
                tol=cfg.exact_tol,
                max_iters=35,
                rcond=cfg.rcond,
            )
            if ok:
                A = analyse(cand, rank, cfg)
                finite = A.max_abs_amplitude <= cfg.coefficient_cap
                cand_generic = genericity_spectrum(
                    cand, rank, cfg, tau=home_generic.tau
                )
                score = generic_basin_score(
                    candidate_soft_nullity=cand_generic.soft_nullity,
                    home_soft_nullity=home_generic.soft_nullity,
                    susceptibility=None,
                    death_distance=A.best_death_distance,
                    max_abs_amplitude=A.max_abs_amplitude,
                    cfg=cfg,
                )
                if finite and score < best_score:
                    best_score = score
                    best_theta = cand
                rows.append(
                    {
                        "trial": trial,
                        "sign": sign,
                        "radius": radius,
                        "obstruction": obstruction,
                        "converged": True,
                        "constraint_residual": rn,
                        "correction_norm": correction,
                        "jacobian_rank": A.jacobian_rank,
                        "tangent_dim": A.tangent_dim,
                        "best_channel": A.best_channel,
                        "best_amplitude": A.best_amplitude,
                        "best_killability": A.best_killability,
                        "best_death_distance": A.best_death_distance,
                        "max_abs_amplitude": A.max_abs_amplitude,
                        "accepted_finite": finite,
                        "genericity_tau": home_generic.tau,
                        "home_soft_nullity": home_generic.soft_nullity,
                        "candidate_soft_nullity": cand_generic.soft_nullity,
                        "soft_nullity_gain": home_generic.soft_nullity - cand_generic.soft_nullity,
                        "generic_branch_score": score,
                    }
                )
            else:
                rows.append(
                    {
                        "trial": trial,
                        "sign": sign,
                        "radius": radius,
                        "obstruction": obstruction,
                        "converged": False,
                        "constraint_residual": rn,
                        "correction_norm": correction,
                        "jacobian_rank": -1,
                        "tangent_dim": -1,
                        "best_channel": -1,
                        "best_amplitude": math.nan,
                        "best_killability": 0.0,
                        "best_death_distance": math.inf,
                        "max_abs_amplitude": math.nan,
                        "accepted_finite": False,
                    }
                )
    return best_theta, rows



def finite_shell_hop_candidates(
    theta: torch.Tensor,
    rank: int,
    cfg: ControllerConfig,
    *,
    radius: float,
    seed: int,
) -> tuple[list[torch.Tensor], list[dict]]:
    """Return all finite exact shell landings for beam search.

    The greedy helper above deliberately chooses one landing.  The beam search
    needs the small set of alternatives so a low-nullity child is not discarded
    merely because another child has slightly better local death distance.
    """
    target = mm_tensor(cfg.n, theta.device)
    gen = torch.Generator(device=theta.device)
    gen.manual_seed(seed)
    rows: list[dict] = []
    candidates: list[torch.Tensor] = []
    home_generic = genericity_spectrum(theta, rank, cfg)

    for trial in range(cfg.hop_trials):
        q, obstruction, _ = obstruction_direction(theta, rank, cfg, generator=gen)
        for sign in (-1.0, 1.0):
            pred = theta + sign * radius * q
            cand, ok, iters, rn, correction = correct_constraints(
                pred,
                target,
                cfg.n,
                rank,
                center=theta,
                radius=radius,
                tol=cfg.exact_tol,
                max_iters=35,
                rcond=cfg.rcond,
            )
            row = {
                "trial": trial,
                "sign": sign,
                "radius": radius,
                "obstruction": obstruction,
                "converged": bool(ok),
                "constraint_residual": rn,
                "correction_norm": correction,
            }
            if not ok:
                rows.append(row)
                continue
            try:
                A = analyse(cand, rank, cfg)
                G = genericity_spectrum(cand, rank, cfg, tau=home_generic.tau)
            except (torch._C._LinAlgError, RuntimeError):
                row["converged"] = False
                row["note"] = "spectral_analysis_failed"
                rows.append(row)
                continue
            finite = A.max_abs_amplitude <= cfg.coefficient_cap
            row.update({
                "finite": finite,
                "tangent_dim": A.tangent_dim,
                "candidate_soft_nullity": G.soft_nullity,
                "soft_nullity_gain": home_generic.soft_nullity - G.soft_nullity,
                "best_death_distance": A.best_death_distance,
                "max_abs_amplitude": A.max_abs_amplitude,
            })
            rows.append(row)
            if finite:
                candidates.append(cand.detach())
    return candidates, rows


def tensor_only_exact_polish(
    theta: torch.Tensor,
    rank: int,
    cfg: ControllerConfig,
    *,
    max_iters: int = 35,
) -> tuple[torch.Tensor, bool, float]:
    """Polish tensor equality when the unit-norm gauge is the stiff part.

    `reconstruct()` already normalizes each factor column, so after a tensor-only
    Gauss-Newton solve we can canonicalize U/V/W to unit norm without changing
    the represented tensor.
    """
    target = mm_tensor(cfg.n, theta.device)
    x = theta.detach().clone()
    for _ in range(max_iters):
        r = residual_vector(x, target, cfg.n, rank).detach()
        rn = float(r.norm())
        if rn <= cfg.exact_tol:
            break
        z = x.detach().clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(
            lambda y: residual_vector(y, target, cfg.n, rank), z, vectorize=True
        ).detach()
        Uj, S, Vh = robust_svd(J, full_matrices=False)
        if len(S) == 0:
            break
        keep = S > cfg.rcond * float(S[0])
        if not bool(keep.any()):
            break
        d = Vh[keep, :].T @ ((Uj[:, keep].T @ (-r)) / S[keep])
        if not bool(torch.isfinite(d).all()):
            break
        base = rn
        alpha = 1.0
        improved = False
        for _ in range(12):
            cand = x + alpha * d
            _, _, _, aa = unpack(cand, cfg.n, rank)
            if float(aa.abs().max()) <= 1.1 * cfg.coefficient_cap:
                cr = float(residual_vector(cand, target, cfg.n, rank).norm())
                if cr < base:
                    x = cand
                    improved = True
                    break
            alpha *= 0.5
        if not improved:
            break

    # Canonical gauge: raw column lengths are irrelevant to reconstruct(), so
    # set them to one explicitly before checking the physical constraints.
    U, V, W, a = unpack(x, cfg.n, rank)
    def unit(X: torch.Tensor) -> torch.Tensor:
        return X / X.norm(dim=0, keepdim=True).clamp_min(1.0e-14)
    x = pack(unit(U), unit(V), unit(W), a).detach()
    rn = float(residual_vector(x, target, cfg.n, rank).norm())
    return x, rn <= cfg.exact_tol, rn



def basin_fingerprint(theta: torch.Tensor, rank: int, cfg: ControllerConfig) -> torch.Tensor:
    """Cheap permutation/sign/gauge-insensitive basin fingerprint.

    We concatenate sorted absolute amplitudes with the spectrum of the Gram
    matrix of normalized rank-one channel tensors.  This is not intended as a
    mathematical complete invariant; it only prevents the exploratory search
    from immediately accepting numerically equivalent basins over and over.
    """
    U, V, W, a = unpack(theta, cfg.n, rank)

    def unit(X: torch.Tensor) -> torch.Tensor:
        return X / X.norm(dim=0, keepdim=True).clamp_min(1.0e-14)

    U, V, W = unit(U), unit(V), unit(W)
    # <u_r⊗v_r⊗w_r, u_s⊗v_s⊗w_s> factorises into three Gram entries.
    G = (U.T @ U) * (V.T @ V) * (W.T @ W)
    evals = torch.linalg.eigvalsh(G).clamp_min(0.0)
    amps = torch.sort(a.abs()).values
    scale = amps.max().clamp_min(1.0)
    return torch.cat([amps / scale, evals, torch.log(scale).reshape(1)]).detach()


def fingerprint_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel():
        return math.inf
    denom = max(1.0, float(a.norm()), float(b.norm()))
    return float((a - b).norm()) / denom


def deletion_susceptibility(
    theta: torch.Tensor,
    rank: int,
    cfg: ControllerConfig,
    *,
    seed: int,
) -> tuple[float, int, list[dict]]:
    """Cheap proxy for distance to *some* rank-(R-1) basin.

    Pick the locally most promising few channels, clamp each directly to zero,
    and give the remaining rank-R parameterization a short bounded relaxation.
    The best tensor residual is the score.  This helper deliberately cannot
    accept a rank drop; DELETE_PROBE/DROP remain the only acceptance path.
    """
    target = mm_tensor(cfg.n, theta.device)
    kills, _ = physical_killability(theta, target, cfg.n, rank, rcond=cfg.rcond)
    ranked = []
    for row in kills:
        k = float(row["killability"])
        amp = abs(float(row["amplitude"]))
        d = amp / max(k, 1.0e-12)
        ranked.append((d, -k, int(row["channel"])))
    ranked.sort()
    channels = [r for _, _, r in ranked[: max(1, cfg.susceptibility_channels)]]

    rows: list[dict] = []
    best_score = math.inf
    best_channel = channels[0]
    gen = torch.Generator(device=theta.device)
    gen.manual_seed(seed)

    for ci, channel in enumerate(channels):
        idx = amp_index(cfg.n, rank, channel)
        x0 = theta.detach().clone()
        # A tiny fixed kick lets an already-generic basin reveal a nearby
        # lower-rank valley without giving this cheap probe a full tunnel budget.
        if cfg.susceptibility_noise > 0:
            noise = cfg.susceptibility_noise * torch.randn(
                x0.shape, dtype=x0.dtype, device=x0.device, generator=gen
            )
            noise[idx] = 0.0
            x0 = x0 + noise
        x0[idx] = 0.0
        x = x0.requires_grad_(True)
        opt = torch.optim.Adam([x], lr=cfg.susceptibility_lr)
        local_best = math.inf
        local_maxamp = math.inf

        for step in range(max(1, cfg.susceptibility_steps)):
            with torch.no_grad():
                x[idx] = 0.0
            # Tensor error is the quantity we actually care about here.  Add a
            # weak unit-column gauge term to stop raw factor norms drifting.
            tensor_r = residual_vector(x, target, cfg.n, rank)
            U, V, W, aa = unpack(x, cfg.n, rank)
            gauge = torch.cat([
                U.square().sum(dim=0) - 1.0,
                V.square().sum(dim=0) - 1.0,
                W.square().sum(dim=0) - 1.0,
            ])
            excess = torch.relu(aa.abs() - cfg.coefficient_cap)
            loss = (tensor_r * tensor_r).sum() + 0.02 * (gauge * gauge).sum() + 100.0 * (excess * excess).sum()
            opt.zero_grad()
            loss.backward()
            if x.grad is not None:
                x.grad[idx] = 0.0
                torch.nn.utils.clip_grad_norm_([x], max_norm=20.0)
            opt.step()
            with torch.no_grad():
                x[idx] = 0.0

            if step % 25 == 0 or step == cfg.susceptibility_steps - 1:
                xd = x.detach()
                rn = float(residual_vector(xd, target, cfg.n, rank).norm())
                _, _, _, aaa = unpack(xd, cfg.n, rank)
                ma = float(aaa.abs().max())
                if math.isfinite(rn) and ma <= 1.1 * cfg.coefficient_cap:
                    local_best = min(local_best, rn)
                    local_maxamp = min(local_maxamp, ma)
                rows.append({
                    "sus_channel": channel,
                    "sus_step": step,
                    "sus_residual": rn,
                    "sus_max_abs_amplitude": ma,
                })
                if rn < 5.0e-3:
                    # Plenty of evidence for ranking; the real delete probe can
                    # decide whether the drop is actually recoverable/exact.
                    break
                if not math.isfinite(rn) or ma > 1.2 * cfg.coefficient_cap:
                    break

        if local_best < best_score:
            best_score = local_best
            best_channel = channel

    return best_score, best_channel, rows


def off_manifold_basin_hop(
    theta: torch.Tensor,
    rank: int,
    cfg: ControllerConfig,
    *,
    radius: float,
    seed: int,
    archive: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor | None, list[dict]]:
    """Tunnel around a failed exact shell hop and recover another exact basin.

    Direct shell Newton can fail when an obstructed tangent direction is only
    infinitesimally feasible.  Here we deliberately allow a bounded physical
    residual while keeping a *soft* finite-radius shell around the current exact
    solution.  The shell pressure is annealed, then an exact corrector is tried
    first with the shell and finally without it.

    Acceptance remains conservative: the landed state must be exact, finite in
    amplitude, and a nontrivial distance from the starting basin.
    """
    target = mm_tensor(cfg.n, theta.device)
    gen = torch.Generator(device=theta.device)
    gen.manual_seed(seed)
    rows: list[dict] = []
    best_theta = None
    best_score = math.inf
    best_susceptibility = math.inf
    best_sus_channel = -1
    best_soft_nullity = math.inf
    home = analyse(theta, rank, cfg)
    home_generic = genericity_spectrum(theta, rank, cfg)
    home_fp = basin_fingerprint(theta, rank, cfg)
    archive = [] if archive is None else archive

    for trial in range(cfg.offhop_trials):
        q, obstruction, _ = obstruction_direction(theta, rank, cfg, generator=gen)
        for sign in (-1.0, 1.0):
            x = (theta + sign * radius * q).detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=cfg.offhop_lr)
            escaped = False

            for stage, shell_weight in enumerate(cfg.offhop_shell_weights):
                for step in range(cfg.offhop_steps_per_stage):
                    c = physical_constraints(x, target, cfg.n, rank)
                    residual_sq = (c * c).sum()
                    dist_sq = ((x - theta) * (x - theta)).sum()
                    # Dimensionless shell error: zero at ||x-home|| = radius.
                    shell = (dist_sq - radius * radius) / max(radius * radius, 1.0e-12)
                    _, _, _, a = unpack(x, cfg.n, rank)
                    excess = torch.relu(a.abs() - cfg.coefficient_cap)
                    cap_penalty = 100.0 * (excess * excess).sum()
                    loss = residual_sq + shell_weight * shell * shell + cap_penalty
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([x], max_norm=25.0)
                    opt.step()

                    if step % 50 == 0 or step == cfg.offhop_steps_per_stage - 1:
                        xd = x.detach()
                        rn = float(physical_constraints(xd, target, cfg.n, rank).norm())
                        dist = float((xd - theta).norm())
                        _, _, _, aa = unpack(xd, cfg.n, rank)
                        ma = float(aa.abs().max())
                        rows.append(
                            {
                                "trial": trial,
                                "sign": sign,
                                "stage": stage,
                                "step": step,
                                "radius": radius,
                                "shell_weight": shell_weight,
                                "obstruction": obstruction,
                                "constraint_residual": rn,
                                "distance_from_home": dist,
                                "max_abs_amplitude": ma,
                                "accepted": False,
                                "landing": "relax",
                            }
                        )
                        if not math.isfinite(rn) or not math.isfinite(ma) or ma > 1.25 * cfg.coefficient_cap:
                            escaped = True
                            break
                if escaped:
                    break
            if escaped:
                continue

            seed_point = x.detach()
            # First try to finish on the requested shell.  This is the cleanest
            # basin-change certificate because it cannot fall back home.
            landed, ok, _, rn, correction = correct_constraints(
                seed_point,
                target,
                cfg.n,
                rank,
                center=theta,
                radius=radius,
                tol=cfg.exact_tol,
                max_iters=cfg.offhop_exact_max_iters,
                rcond=cfg.rcond,
            )
            landing = "exact_shell"

            # If the exact shell remains singular, release it only after the soft
            # tunnel has moved far away.  A minimum distance check below prevents
            # an uninteresting return to the home solution.
            if not ok:
                # Start the free correction from the best point reached by the
                # shell corrector, not from the raw soft-relaxation seed.
                free_seed = landed if bool(torch.isfinite(landed).all()) else seed_point
                landed, ok, _, rn, correction_free = correct_constraints(
                    free_seed,
                    target,
                    cfg.n,
                    rank,
                    tol=cfg.exact_tol,
                    max_iters=cfg.offhop_exact_max_iters,
                    rcond=cfg.rcond,
                )
                correction += correction_free
                landing = "exact_free"

            if not ok:
                polished, tensor_ok, tensor_rn = tensor_only_exact_polish(landed, rank, cfg)
                if tensor_ok:
                    landed = polished
                    ok = True
                    rn = tensor_rn
                    landing = "tensor_polish"

            if not ok:
                rows.append(
                    {
                        "trial": trial, "sign": sign, "stage": -1, "step": -1,
                        "radius": radius, "shell_weight": 0.0, "obstruction": obstruction,
                        "constraint_residual": rn, "distance_from_home": float((landed-theta).norm()),
                        "max_abs_amplitude": math.nan, "accepted": False, "landing": landing,
                    }
                )
                continue

            try:
                A = analyse(landed, rank, cfg)
                cand_generic = genericity_spectrum(
                    landed, rank, cfg, tau=home_generic.tau
                )
            except (torch._C._LinAlgError, RuntimeError):
                continue
            dist = float((landed - theta).norm())
            finite = A.max_abs_amplitude <= cfg.coefficient_cap
            moved = dist >= cfg.offhop_min_distance_fraction * radius

            fp = basin_fingerprint(landed, rank, cfg)
            novelty_home = fingerprint_distance(fp, home_fp)
            novelty_archive = min((fingerprint_distance(fp, old) for old in archive if old.numel() == fp.numel()), default=math.inf)
            novel = novelty_archive >= cfg.archive_novelty_tol

            susceptibility, sus_channel, sus_rows = deletion_susceptibility(
                landed, rank, cfg, seed=seed + 1000003 * trial + (0 if sign < 0 else 1)
            )
            for sr in sus_rows:
                sr.update({
                    "trial": trial, "sign": sign, "stage": -2, "step": sr.pop("sus_step"),
                    "radius": radius, "shell_weight": 0.0, "obstruction": obstruction,
                    "landing": "susceptibility_probe",
                })
                rows.append(sr)

            # Generic global strategy at every rank: first reward opening
            # excess Jacobian null modes, while the logarithmic deletion term
            # becomes dominant automatically when a rank-drop basin is close.
            score = generic_basin_score(
                candidate_soft_nullity=cand_generic.soft_nullity,
                home_soft_nullity=home_generic.soft_nullity,
                susceptibility=susceptibility,
                death_distance=A.best_death_distance,
                max_abs_amplitude=A.max_abs_amplitude,
                cfg=cfg,
            )
            accepted = finite and moved and novel and math.isfinite(susceptibility)
            rows.append(
                {
                    "trial": trial, "sign": sign, "stage": -1, "step": -1,
                    "radius": radius, "shell_weight": 0.0, "obstruction": obstruction,
                    "constraint_residual": A.residual, "distance_from_home": dist,
                    "max_abs_amplitude": A.max_abs_amplitude, "accepted": accepted,
                    "landing": landing, "best_channel": A.best_channel,
                    "best_amplitude": A.best_amplitude, "best_killability": A.best_killability,
                    "best_death_distance": A.best_death_distance,
                    "delete_susceptibility": susceptibility, "susceptibility_channel": sus_channel,
                    "novelty_from_home": novelty_home, "novelty_from_archive": novelty_archive,
                    "tangent_dim": A.tangent_dim, "home_tangent_dim": home.tangent_dim,
                    "genericity_tau": home_generic.tau,
                    "home_soft_nullity": home_generic.soft_nullity,
                    "candidate_soft_nullity": cand_generic.soft_nullity,
                    "soft_nullity_gain": home_generic.soft_nullity - cand_generic.soft_nullity,
                    "correction_norm": correction, "branch_score": score,
                }
            )
            if accepted and score < best_score:
                best_score = score
                best_theta = landed
                best_susceptibility = susceptibility
                best_sus_channel = sus_channel
                best_soft_nullity = cand_generic.soft_nullity

    if best_theta is not None:
        rows.append({
            "trial": -1, "sign": 0.0, "stage": -3, "step": -1,
            "radius": radius, "landing": "selected_summary", "accepted": True,
            "delete_susceptibility": best_susceptibility,
            "susceptibility_channel": best_sus_channel,
            "genericity_tau": home_generic.tau,
            "home_soft_nullity": home_generic.soft_nullity,
            "candidate_soft_nullity": best_soft_nullity,
            "soft_nullity_gain": home_generic.soft_nullity - best_soft_nullity,
            "branch_score": best_score,
        })
    return best_theta, rows


def off_manifold_delete_probe(
    theta: torch.Tensor,
    rank: int,
    channel: int,
    cfg: ControllerConfig,
    *,
    seed: int,
) -> tuple[torch.Tensor, list[dict], bool]:
    """Clamp a channel through the wall and let the other channels reorganise."""
    torch.manual_seed(seed)
    target = mm_tensor(cfg.n, theta.device)
    idx = amp_index(cfg.n, rank, channel)
    _, _, _, a0 = unpack(theta, cfg.n, rank)
    initial = float(a0[channel])
    x = theta.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=cfg.delete_lr)
    rows: list[dict] = []

    stages = list(cfg.clamp_fractions)
    for stage_idx, frac in enumerate(stages):
        target_amp = initial * frac
        stage_steps = cfg.delete_final_steps if frac == 0.0 else cfg.delete_steps_per_stage
        best_stage = math.inf
        stale = 0
        for step in range(stage_steps):
            with torch.no_grad():
                x[idx] = target_amp
            c = physical_constraints(x, target, cfg.n, rank)
            residual_sq = (c * c).sum()
            _, _, _, a = unpack(x, cfg.n, rank)
            excess = torch.relu(a.abs() - cfg.coefficient_cap)
            cap_penalty = 100.0 * (excess * excess).sum()
            loss = residual_sq + cap_penalty
            opt.zero_grad()
            loss.backward()
            if x.grad is not None:
                x.grad[idx] = 0.0
                torch.nn.utils.clip_grad_norm_([x], max_norm=20.0)
            opt.step()
            with torch.no_grad():
                x[idx] = target_amp

            if step % 50 == 0 or step == stage_steps - 1:
                rn = float(residual_vector(x.detach(), target, cfg.n, rank).norm())
                _, _, _, aa = unpack(x.detach(), cfg.n, rank)
                ma = float(aa.abs().max())
                rows.append(
                    {
                        "stage": stage_idx,
                        "fraction": frac,
                        "step": step,
                        "target_amplitude": target_amp,
                        "tensor_residual": rn,
                        "max_abs_amplitude": ma,
                    }
                )
                if rn + 1.0e-7 < best_stage:
                    best_stage = rn
                    stale = 0
                else:
                    stale += 1
                # At the zero stage we care about the basin transition, not
                # endlessly polishing with Adam once a good lower-rank basin is
                # clearly found.
                if frac == 0.0 and rn < cfg.delete_accept_residual:
                    return x.detach(), rows, True
                if frac == 0.0 and stale > 18 and rn > 0.25:
                    # Add a small symmetry-breaking kick rather than burning the
                    # entire budget at the obvious missing-product residual.
                    with torch.no_grad():
                        noise = 1.0e-3 * torch.randn_like(x)
                        noise[idx] = 0.0
                        x.add_(noise)
                    stale = 0

    final_r = float(residual_vector(x.detach(), target, cfg.n, rank).norm())
    return x.detach(), rows, final_r < cfg.delete_accept_residual


def drop_and_correct(
    theta: torch.Tensor,
    rank: int,
    channel: int,
    cfg: ControllerConfig,
) -> tuple[torch.Tensor | None, float, float]:
    U, V, W, a = unpack(theta, cfg.n, rank)
    keep = [r for r in range(rank) if r != channel]
    q = pack(U[:, keep], V[:, keep], W[:, keep], a[keep])
    target = mm_tensor(cfg.n, theta.device)
    corrected, ok, _, rn, _ = correct_constraints(
        q,
        target,
        cfg.n,
        rank - 1,
        tol=cfg.delete_exact_tol,
        max_iters=45,
        rcond=max(cfg.rcond, 1.0e-9),
    )
    if not ok:
        # A tensor-only GN polish can sometimes finish when the physical gauge
        # constraints are the numerically stiff part.  Re-normalise through the
        # representation afterwards.
        x = q.detach().clone()
        for _ in range(30):
            r = residual_vector(x, target, cfg.n, rank - 1).detach()
            if float(r.norm()) <= cfg.delete_exact_tol:
                break
            z = x.detach().clone().requires_grad_(True)
            J = torch.autograd.functional.jacobian(
                lambda y: residual_vector(y, target, cfg.n, rank - 1), z, vectorize=True
            ).detach()
            Uj, S, Vh = robust_svd(J, full_matrices=False)
            keep_s = S > max(cfg.rcond, 1e-9) * float(S[0])
            if not bool(keep_s.any()):
                break
            d = Vh[keep_s, :].T @ ((Uj[:, keep_s].T @ (-r)) / S[keep_s])
            alpha = 1.0
            base = float(r.norm())
            improved = False
            for _ in range(10):
                cand = x + alpha * d
                if float(residual_vector(cand, target, cfg.n, rank - 1).norm()) < base:
                    x = cand
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                break
        corrected = x
        rn = float(residual_vector(corrected, target, cfg.n, rank - 1).norm())
        ok = rn <= cfg.delete_exact_tol

    if not ok:
        return None, rn, math.inf
    _, _, _, aa = unpack(corrected, cfg.n, rank - 1)
    return corrected, rn, float(aa.abs().max())


def run_controller(
    start: torch.Tensor,
    start_rank: int,
    cfg: ControllerConfig,
    out: Path,
) -> tuple[torch.Tensor, int, list[Transition]]:
    out.mkdir(parents=True, exist_ok=True)
    theta = start.detach().clone()
    rank = start_rank
    phase = Phase.CONTINUE_TO_WALL
    hop_count = 0
    radius = cfg.hop_radius
    selected_channel = -1
    transitions: list[Transition] = []
    offhop_failures = 0
    basin_archive: list[torch.Tensor] = [basin_fingerprint(theta, rank, cfg)]

    save_theta(out / f"rank{rank}_initial.pt", theta, rank, phase="initial")

    for cycle in range(cfg.max_cycles):
        # Persist the current state *before* any spectral diagnostics.  This
        # makes ill-conditioned Jacobians recoverable numerical events rather
        # than lost exploratory trajectories.
        save_theta(out / "latest_state.pt", theta, rank, phase=phase.value, cycle=cycle)
        cheap_residual = float(residual_vector(theta, mm_tensor(cfg.n, theta.device), cfg.n, rank).norm())
        if cheap_residual <= max(cfg.exact_tol, 1.0e-9):
            save_theta(out / "latest_exact.pt", theta, rank, phase=phase.value, cycle=cycle)

        A0 = analyse(theta, rank, cfg)
        print(
            f"cycle={cycle:02d} phase={phase.value:>17s} rank={rank} "
            f"res={A0.residual:.2e} best=r{A0.best_channel} "
            f"a={A0.best_amplitude:+.4f} K={A0.best_killability:.3f} "
            f"D={A0.best_death_distance:.3f} null={A0.tangent_dim}"
        )

        if rank <= cfg.goal_rank:
            phase = Phase.DONE
            break

        if phase == Phase.CONTINUE_TO_WALL:
            selected_channel = A0.best_channel
            before_amp = A0.best_amplitude
            theta2, rows, status = continue_to_wall(theta, rank, selected_channel, cfg)
            write_csv(out / f"cycle_{cycle:02d}_continue.csv", rows)
            A1 = analyse(theta2, rank, cfg)
            transitions.append(
                Transition(cycle, phase.value, rank, selected_channel, A0.residual, A1.residual,
                           before_amp, float(unpack(theta2,cfg.n,rank)[3][selected_channel]),
                           A0.best_killability,
                           float(physical_killability(theta2,mm_tensor(cfg.n),cfg.n,rank,rcond=cfg.rcond)[0][selected_channel]["killability"]),
                           A1.max_abs_amplitude, status)
            )
            theta = theta2
            save_theta(out / f"cycle_{cycle:02d}_wall.pt", theta, rank, phase=phase.value, status=status)
            if status == "zero_reached":
                phase = Phase.DROP
            else:
                phase = Phase.HOP

        elif phase == Phase.HOP:
            landed, rows = finite_shell_hop(
                theta, rank, cfg, radius=radius, seed=cfg.seed + 1009 * cycle
            )
            write_csv(out / f"cycle_{cycle:02d}_hop.csv", rows)
            if landed is None:
                radius *= cfg.hop_radius_growth
                if radius > cfg.hop_max_radius:
                    transitions.append(
                        Transition(cycle, phase.value, rank, -1, A0.residual, A0.residual,
                                   math.nan, math.nan, A0.best_killability, A0.best_killability,
                                   A0.max_abs_amplitude,
                                   f"exact shell exhausted at radius={radius:.3f}; tunnel next")
                    )
                    # Exact Newton-shell continuation has exhausted this local
                    # sector.  That is a dynamical event, not terminal failure.
                    phase = Phase.OFF_MANIFOLD_HOP
                    radius = cfg.hop_max_radius
                    continue
                transitions.append(
                    Transition(cycle, phase.value, rank, -1, A0.residual, A0.residual,
                               math.nan, math.nan, A0.best_killability, A0.best_killability,
                               A0.max_abs_amplitude, f"no shell landing; radius -> {radius:.3f}")
                )
                continue
            A1 = analyse(landed, rank, cfg)
            transitions.append(
                Transition(cycle, phase.value, rank, A1.best_channel, A0.residual, A1.residual,
                           A0.best_amplitude, A1.best_amplitude, A0.best_killability,
                           A1.best_killability, A1.max_abs_amplitude,
                           f"exact shell hop radius={radius:.3f}")
            )
            theta = landed
            basin_archive.append(basin_fingerprint(theta, rank, cfg))
            hop_count += 1
            save_theta(out / f"cycle_{cycle:02d}_hop.pt", theta, rank, phase=phase.value, radius=radius)
            # Periodically challenge the wall.  Between probes, let exact
            # continuation exploit any mobility created by the hop.
            phase = Phase.DELETE_PROBE if hop_count % cfg.delete_every_hops == 0 else Phase.CONTINUE_TO_WALL

        elif phase == Phase.OFF_MANIFOLD_HOP:
            landed, rows = off_manifold_basin_hop(
                theta, rank, cfg, radius=radius, seed=cfg.seed + 15401 * cycle,
                archive=basin_archive,
            )
            write_csv(out / f"cycle_{cycle:02d}_offhop.csv", rows)
            if landed is None:
                offhop_failures += 1
                transitions.append(
                    Transition(cycle, phase.value, rank, -1, A0.residual, A0.residual,
                               math.nan, math.nan, A0.best_killability, A0.best_killability,
                               A0.max_abs_amplitude,
                               f"off-manifold tunnel failed ({offhop_failures}); reset exact hops")
                )
                # Do not terminate the exploratory run.  Change the random
                # direction on the next cycle and rebuild from small exact hops.
                radius = cfg.hop_radius
                phase = Phase.HOP
                continue

            A1 = analyse(landed, rank, cfg)
            transitions.append(
                Transition(cycle, phase.value, rank, A1.best_channel, A0.residual, A1.residual,
                           A0.best_amplitude, A1.best_amplitude, A0.best_killability,
                           A1.best_killability, A1.max_abs_amplitude,
                           f"accepted off-manifold basin hop radius={radius:.3f}")
            )
            theta = landed
            basin_archive.append(basin_fingerprint(theta, rank, cfg))
            offhop_failures = 0
            radius = cfg.hop_radius
            selected_rows = [r for r in rows if r.get("landing") == "selected_summary"]
            selected_channel = int(selected_rows[-1]["susceptibility_channel"]) if selected_rows else -1
            save_theta(out / f"cycle_{cycle:02d}_offhop.pt", theta, rank,
                       phase=phase.value, source="soft_shell_tunnel",
                       susceptibility_channel=selected_channel)
            # The tunnel was selected specifically because this channel looked
            # close to a lower-rank basin.  Challenge it immediately before
            # local exact motion can wander away from that basin.
            phase = Phase.DELETE_PROBE

        elif phase == Phase.DELETE_PROBE:
            Aprobe = analyse(theta, rank, cfg)
            if selected_channel < 0 or selected_channel >= rank:
                selected_channel = Aprobe.best_channel
            probe, rows, promising = off_manifold_delete_probe(
                theta, rank, selected_channel, cfg, seed=cfg.seed + 7919 * cycle
            )
            write_csv(out / f"cycle_{cycle:02d}_delete_probe.csv", rows)
            probe_res = float(residual_vector(probe, mm_tensor(cfg.n), cfg.n, rank).norm())
            _, _, _, ap = unpack(probe, cfg.n, rank)
            transitions.append(
                Transition(cycle, phase.value, rank, selected_channel, A0.residual, probe_res,
                           Aprobe.best_amplitude, float(ap[selected_channel]), Aprobe.best_killability,
                           0.0, float(ap.abs().max()), "promising" if promising else "no basin transition")
            )
            save_theta(out / f"cycle_{cycle:02d}_probe.pt", probe, rank, phase=phase.value, promising=promising)
            if promising:
                theta = probe
                phase = Phase.DROP
            else:
                # Return to the exact point, then make another finite hop.
                selected_channel = -1
                phase = Phase.HOP
                radius = min(cfg.hop_max_radius, radius * cfg.hop_radius_growth)

        elif phase == Phase.DROP:
            if selected_channel < 0:
                selected_channel = analyse(theta, rank, cfg).best_channel
            dropped, rn, maxamp = drop_and_correct(theta, rank, selected_channel, cfg)
            if dropped is None or maxamp > cfg.coefficient_cap:
                transitions.append(
                    Transition(cycle, phase.value, rank, selected_channel, A0.residual, rn,
                               float(unpack(theta,cfg.n,rank)[3][selected_channel]), math.nan,
                               A0.best_killability, 0.0, maxamp, "rank drop correction failed")
                )
                phase = Phase.HOP
                continue
            old_rank = rank
            theta = dropped
            rank -= 1
            A1 = analyse(theta, rank, cfg)
            transitions.append(
                Transition(cycle, phase.value, old_rank, selected_channel, A0.residual, A1.residual,
                           0.0, math.nan, A0.best_killability, A1.best_killability,
                           A1.max_abs_amplitude, f"accepted rank {old_rank}->{rank}")
            )
            print(f"*** rank drop {old_rank}->{rank}: residual={A1.residual:.3e} max|a|={A1.max_abs_amplitude:.3f}")
            save_theta(out / f"rank{rank}_blind.pt", theta, rank, phase=phase.value, parent_rank=old_rank)
            phase = Phase.CONTINUE_TO_WALL
            hop_count = 0
            radius = cfg.hop_radius
            selected_channel = -1
            offhop_failures = 0
            basin_archive = [basin_fingerprint(theta, rank, cfg)]

    Afinal = analyse(theta, rank, cfg)
    payload = {
        "config": asdict(cfg),
        "final_rank": rank,
        "final_analysis": asdict(Afinal),
        "phase": phase.value,
        "transitions": [asdict(t) for t in transitions],
    }
    with (out / "state_machine_summary.json").open("w") as f:
        json.dump(payload, f, indent=2)
    write_csv(out / "state_machine_transitions.csv", [asdict(t) for t in transitions])
    save_theta(out / "final.pt", theta, rank, phase=phase.value)
    return theta, rank, transitions



def _preselect_exact_children(
    children: list[BasinState],
    limit: int,
) -> list[BasinState]:
    """Cheap preselection before expensive susceptibility probes."""
    if not children or limit <= 0:
        return []
    # Susceptibility is inf for provisional children.  Select specialists in
    # genericity and local death distance, then fill by their simple rank sum.
    chosen: list[BasinState] = []
    seen: set[int] = set()
    for state in (
        min(children, key=lambda s: (s.soft_nullity, s.basin_id)),
        min(children, key=lambda s: (s.death_distance, s.basin_id)),
        min(children, key=lambda s: (s.max_abs_amplitude, s.basin_id)),
    ):
        if state.basin_id not in seen and len(chosen) < limit:
            chosen.append(state)
            seen.add(state.basin_id)
    if len(chosen) < limit:
        ng = _rank_positions(children, lambda s: s.soft_nullity)
        nd = _rank_positions(children, lambda s: s.death_distance)
        for state in sorted(children, key=lambda s: (ng[s.basin_id] + nd[s.basin_id], s.basin_id)):
            if state.basin_id not in seen and len(chosen) < limit:
                chosen.append(state)
                seen.add(state.basin_id)
    return chosen


def _polish_retained_state(
    state: BasinState,
    rank: int,
    cfg: ControllerConfig,
    *,
    tau: float,
) -> BasinState:
    """Tighten tensor equality for a retained basin without changing its search identity.

    Susceptibility is intentionally preserved: re-running its noisy Adam probe
    solely because of a ~1e-12 Gauss-Newton cleanup would waste compute and add
    noise.  Geometry and fingerprint are recomputed from the polished point.
    """
    polish_cfg = replace(cfg, exact_tol=min(cfg.exact_tol, cfg.beam_polish_tol))
    polished, ok, rn = tensor_only_exact_polish(
        state.theta, rank, polish_cfg, max_iters=cfg.beam_polish_max_iters
    )
    if not ok and rn >= state.analysis.residual:
        return state
    try:
        A = analyse(polished, rank, cfg)
        G = genericity_spectrum(polished, rank, cfg, tau=tau)
    except (torch._C._LinAlgError, RuntimeError):
        return state
    if A.max_abs_amplitude > cfg.coefficient_cap:
        return state
    return replace(
        state,
        theta=polished.detach().clone(),
        analysis=A,
        soft_nullity=G.soft_nullity,
        max_abs_amplitude=A.max_abs_amplitude,
        fingerprint=basin_fingerprint(polished, rank, cfg),
    )


def _polish_frontier(
    frontier: list[BasinState],
    rank: int,
    cfg: ControllerConfig,
    *,
    tau: float,
) -> list[BasinState]:
    return [_polish_retained_state(s, rank, cfg, tau=tau) for s in frontier]


def _save_beam_champions(out: Path, frontier: list[BasinState], generation: int):
    if not frontier:
        return
    champions = {
        "best_genericity.pt": min(frontier, key=lambda s: (s.soft_nullity, s.basin_id)),
        "best_delete_susceptibility.pt": min(frontier, key=lambda s: (s.susceptibility, s.basin_id)),
        "best_death_distance.pt": min(frontier, key=lambda s: (s.death_distance, s.basin_id)),
    }
    for name, state in champions.items():
        save_theta(
            out / name,
            state.theta,
            state.rank,
            basin_id=state.basin_id,
            generation=generation,
            source=state.source,
            soft_nullity=state.soft_nullity,
            susceptibility=state.susceptibility,
            death_distance=state.death_distance,
            expansion_count=state.expansion_count,
            last_expanded_generation=state.last_expanded_generation,
        )


def run_beam_controller(
    start: torch.Tensor,
    start_rank: int,
    cfg: ControllerConfig,
    out: Path,
) -> tuple[torch.Tensor, int, list[dict]]:
    """Endpoint-free Pareto beam search over exact algorithm basins.

    The local differential machinery is unchanged.  The only conceptual change
    from ``run_controller`` is global policy: we retain a tiny Pareto frontier
    of exact basins rather than overwriting the current state after every hop.
    This makes non-monotone geometry survivable without turning the experiment
    into a large population search.
    """
    out.mkdir(parents=True, exist_ok=True)
    rank = start_rank
    next_id = 0
    history: list[dict] = []
    expansion_log: list[dict] = []
    archive: list[torch.Tensor] = []

    def fresh_state(
        theta: torch.Tensor,
        *,
        parent_id: int,
        generation: int,
        source: str,
        tau: float,
        probe: bool = True,
        seed_offset: int = 0,
    ) -> BasinState:
        nonlocal next_id
        state = make_basin_state(
            theta,
            rank,
            cfg,
            basin_id=next_id,
            parent_id=parent_id,
            generation=generation,
            source=source,
            tau=tau,
            seed=cfg.seed + 104729 * generation + 1009 * next_id + seed_offset,
            probe_susceptibility=probe,
        )
        next_id += 1
        return state

    # A single tau is used for comparison within each rank.  After a drop the
    # new rank gets its own intrinsic scale; no target nullity is encoded.
    rank_tau = genericity_spectrum(start, rank, cfg).tau
    initial = fresh_state(
        start.detach().clone(), parent_id=-1, generation=0, source="initial", tau=rank_tau
    )
    frontier = [initial]
    archive.append(initial.fingerprint)
    save_theta(out / f"rank{rank}_beam_initial.pt", initial.theta, rank, basin_id=initial.basin_id)
    _save_beam_champions(out, frontier, 0)

    for generation in range(cfg.max_cycles):
        # Persist the whole frontier before doing any expensive spectral work.
        for i, state in enumerate(frontier):
            save_theta(
                out / f"frontier_g{generation:02d}_{i:02d}_id{state.basin_id}.pt",
                state.theta,
                rank,
                basin_id=state.basin_id,
                parent_id=state.parent_id,
                generation=generation,
                source=state.source,
            )
            history.append(basin_state_row(state, accepted=True, note="frontier"))

        ordered = beam_priority_order(frontier)
        summary = " | ".join(
            f"id{s.basin_id}:N={s.soft_nullity:.1f} E={s.susceptibility:.3g} D={s.death_distance:.2f}"
            for s in ordered
        )
        print(f"beam gen={generation:02d} rank={rank} frontier={len(frontier)} | {summary}")

        if rank <= cfg.goal_rank:
            break

        # Periodically challenge the most deletion-susceptible frontier state.
        delete_now = (
            generation % max(1, cfg.beam_delete_every) == 0
            or min(s.susceptibility for s in frontier) <= cfg.beam_delete_trigger
        )
        rank_dropped = False
        if delete_now:
            for state in sorted(frontier, key=lambda s: (s.susceptibility, s.death_distance))[
                : max(1, cfg.beam_delete_probe_states)
            ]:
                channel = state.susceptibility_channel
                probe, rows, promising = off_manifold_delete_probe(
                    state.theta,
                    rank,
                    channel,
                    cfg,
                    seed=cfg.seed + 7919 * generation + state.basin_id,
                )
                write_csv(out / f"beam_g{generation:02d}_id{state.basin_id}_delete.csv", rows)
                probe_res = float(residual_vector(probe, mm_tensor(cfg.n), cfg.n, rank).norm())
                history.append({
                    **basin_state_row(state, accepted=promising, note="full_delete_probe"),
                    "probe_residual": probe_res,
                    "probe_channel": channel,
                })
                if not promising:
                    continue
                dropped, rn, maxamp = drop_and_correct(probe, rank, channel, cfg)
                if dropped is None or maxamp > cfg.coefficient_cap:
                    continue

                old_rank = rank
                rank -= 1
                print(
                    f"*** beam rank drop {old_rank}->{rank}: residual={rn:.3e} "
                    f"max|a|={maxamp:.3f} parent=id{state.basin_id} channel=r{channel}"
                )
                save_theta(
                    out / f"rank{rank}_beam_blind.pt",
                    dropped,
                    rank,
                    parent_rank=old_rank,
                    parent_basin_id=state.basin_id,
                    deleted_channel=channel,
                )
                # Reset global geometry only because the parameter space itself
                # changed dimension.  The same generic strategy then repeats.
                rank_tau = genericity_spectrum(dropped, rank, cfg).tau
                frontier = [
                    fresh_state(
                        dropped,
                        parent_id=state.basin_id,
                        generation=generation + 1,
                        source=f"drop_{old_rank}_to_{rank}",
                        tau=rank_tau,
                    )
                ]
                archive = [frontier[0].fingerprint]
                _save_beam_champions(out, frontier, generation + 1)
                rank_dropped = True
                break
            if rank_dropped:
                if rank <= cfg.goal_rank:
                    break
                continue

        schedule = specialist_expansion_schedule(frontier, generation, cfg)
        role_summary = " | ".join(
            f"id{state.basin_id}:{'+'.join(roles)}(x{state.expansion_count})"
            for state, roles in schedule
        )
        print(f"beam expand gen={generation:02d}: {role_summary}")
        new_children: list[BasinState] = []

        for parent, roles in schedule:
            parent.expansion_count += 1
            parent.last_expanded_generation = generation
            expansion_log.append({
                "generation": generation,
                "rank": rank,
                "basin_id": parent.basin_id,
                "parent_id": parent.parent_id,
                "roles": "+".join(roles),
                "expansion_count": parent.expansion_count,
                "soft_nullity": parent.soft_nullity,
                "susceptibility": parent.susceptibility,
                "death_distance": parent.death_distance,
                "source": parent.source,
            })
            history.append(basin_state_row(parent, accepted=True, note=f"expanded:{'+'.join(roles)}"))

            # Local exact exploitation first.  Every specialist gets this cheap
            # step; the expensive global operator below is role-specific.
            wall_theta, wall_rows, wall_status = continue_to_wall(
                parent.theta, rank, parent.analysis.best_channel, cfg
            )
            write_csv(out / f"beam_g{generation:02d}_id{parent.basin_id}_continue.csv", wall_rows)
            provisional: list[BasinState] = []
            try:
                wall_A = analyse(wall_theta, rank, cfg)
                if wall_A.residual <= max(cfg.exact_tol * 100.0, 1.0e-8) and wall_A.max_abs_amplitude <= cfg.coefficient_cap:
                    state = fresh_state(
                        wall_theta,
                        parent_id=parent.basin_id,
                        generation=generation + 1,
                        source=f"wall:{wall_status}",
                        tau=rank_tau,
                        probe=False,
                    )
                    provisional.append(state)
            except (torch._C._LinAlgError, RuntimeError):
                pass

            # Branch over several exact shell landings rather than selecting one.
            exact_thetas, exact_rows = finite_shell_hop_candidates(
                wall_theta,
                rank,
                cfg,
                radius=cfg.hop_radius,
                seed=cfg.seed + 1009 * generation + 37 * parent.basin_id,
            )
            write_csv(out / f"beam_g{generation:02d}_id{parent.basin_id}_exact_hops.csv", exact_rows)
            for k, cand in enumerate(exact_thetas):
                try:
                    st = fresh_state(
                        cand,
                        parent_id=parent.basin_id,
                        generation=generation + 1,
                        source="exact_hop",
                        tau=rank_tau,
                        probe=False,
                        seed_offset=k,
                    )
                except (torch._C._LinAlgError, RuntimeError):
                    continue
                novel, novelty = fingerprint_is_novel(st.fingerprint, archive, cfg.archive_novelty_tol)
                if novel:
                    provisional.append(st)

            # Only susceptibility-probe a few geometrically distinct exact
            # children.  This keeps beam branching much cheaper than evaluating
            # every shell landing with Adam.
            shortlisted = _preselect_exact_children(provisional, cfg.beam_exact_children)
            realized: list[BasinState] = []
            for st in shortlisted:
                full = fresh_state(
                    st.theta,
                    parent_id=st.parent_id,
                    generation=st.generation,
                    source=st.source,
                    tau=rank_tau,
                    probe=True,
                )
                novel, novelty = fingerprint_is_novel(full.fingerprint, archive, cfg.archive_novelty_tol)
                # The wall child may be close in fingerprint to its parent but is
                # still worth keeping when it materially improves local death.
                materially_better = (
                    full.soft_nullity < parent.soft_nullity - 1.0e-3
                    or full.susceptibility < parent.susceptibility * 0.98
                    or full.death_distance < parent.death_distance * 0.98
                )
                if novel or materially_better:
                    realized.append(full)
                    archive.append(full.fingerprint)

            new_children.extend(realized)

            best_soft = min((c.soft_nullity for c in realized), default=math.inf)
            best_sus = min((c.susceptibility for c in realized), default=math.inf)
            soft_gain = parent.soft_nullity - best_soft
            susceptibility_gain = parent.susceptibility - best_sus

            # Specialist-aware global operator:
            #   genericity -> proactively spend more tunnel budget opening modes;
            #   deletion   -> tunnel when local rearrangement did not improve E;
            #   death      -> tunnel only after the local wall/exact-hop stalls;
            #   explore    -> retain the old mixed fallback.
            force_genericity = "genericity" in roles
            deletion_stalled = (
                "deletion" in roles
                and (not math.isfinite(best_sus) or susceptibility_gain < 0.02 * max(parent.susceptibility, 1.0e-6))
            )
            should_offhop = force_genericity or deletion_stalled or soft_gain < cfg.beam_min_soft_gain
            if should_offhop:
                offhop_children = (
                    max(cfg.beam_genericity_offhop_children, cfg.beam_offhop_children)
                    if force_genericity
                    else max(1, cfg.beam_offhop_children)
                )
                for oi in range(offhop_children):
                    tunnel_cfg = replace(cfg, offhop_trials=1)
                    landed, off_rows = off_manifold_basin_hop(
                        parent.theta,
                        rank,
                        tunnel_cfg,
                        radius=min(cfg.hop_max_radius, cfg.hop_radius * cfg.hop_radius_growth),
                        seed=cfg.seed + 15401 * generation + 503 * parent.basin_id + oi,
                        archive=archive,
                    )
                    write_csv(out / f"beam_g{generation:02d}_id{parent.basin_id}_offhop_{oi}.csv", off_rows)
                    if landed is None:
                        continue
                    try:
                        full = fresh_state(
                            landed,
                            parent_id=parent.basin_id,
                            generation=generation + 1,
                            source="off_manifold_hop",
                            tau=rank_tau,
                            probe=True,
                            seed_offset=10000 + oi,
                        )
                    except (torch._C._LinAlgError, RuntimeError):
                        continue
                    novel, _ = fingerprint_is_novel(full.fingerprint, archive, cfg.archive_novelty_tol)
                    if novel:
                        new_children.append(full)
                        archive.append(full.fingerprint)

        # Parents stay in the pool.  This is the central difference from the
        # greedy controller: a promising basin cannot be erased by one bad hop.
        pool = frontier + new_children
        frontier = pareto_beam_prune(pool, cfg.beam_width)
        # Archive retained basins at a consistent tensor residual so Jacobian
        # spectra are compared on geometry rather than correction tolerance.
        frontier = _polish_frontier(frontier, rank, cfg, tau=rank_tau)
        for state in frontier:
            history.append(basin_state_row(state, accepted=True, note="pareto_kept_polished"))
            archive.append(state.fingerprint)
        _save_beam_champions(out, frontier, generation + 1)
        write_csv(out / "beam_history.csv", history)
        write_csv(out / "beam_expansions.csv", expansion_log)

        if not new_children:
            print("beam: no novel child this generation; retaining frontier and changing seeds")

    # Final state is the balanced priority leader, but all frontier states and
    # champion checkpoints remain on disk.
    final_state = beam_priority_order(frontier)[0]
    summary = {
        "config": asdict(cfg),
        "final_rank": rank,
        "frontier": [basin_state_row(s, accepted=True, note="final_frontier") for s in frontier],
        "final_basin_id": final_state.basin_id,
        "expansions": expansion_log,
    }
    with (out / "beam_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    write_csv(out / "beam_history.csv", history)
    write_csv(out / "beam_expansions.csv", expansion_log)
    save_theta(out / "final.pt", final_state.theta, rank, basin_id=final_state.basin_id, source=final_state.source)
    return final_state.theta, rank, history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=Path, default=DEFAULT_START)
    p.add_argument("--goal-rank", type=int, default=23)
    p.add_argument("--out", type=Path, default=Path("runs/autonomous_state_machine"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-cycles", type=int, default=16)
    p.add_argument("--hop-trials", type=int, default=5)
    p.add_argument("--hop-obstruction-trials", type=int, default=6)
    p.add_argument("--hop-radius", type=float, default=2.5)
    p.add_argument("--offhop-trials", type=int, default=3)
    p.add_argument("--offhop-steps", type=int, default=350)
    p.add_argument("--offhop-lr", type=float, default=1.5e-2)
    p.add_argument("--delete-every-hops", type=int, default=2)
    p.add_argument("--delete-steps", type=int, default=700)
    p.add_argument("--delete-final-steps", type=int, default=2500)
    p.add_argument("--coefficient-cap", type=float, default=8.0)
    p.add_argument("--susceptibility-channels", type=int, default=3)
    p.add_argument("--susceptibility-steps", type=int, default=140)
    p.add_argument("--susceptibility-lr", type=float, default=2.0e-2)
    p.add_argument("--archive-novelty-tol", type=float, default=2.0e-3)
    p.add_argument("--genericity-weight", type=float, default=6.0)
    p.add_argument("--genericity-deletion-weight", type=float, default=1.0)
    p.add_argument("--genericity-deletion-scale", type=float, default=1.0e-1)
    p.add_argument("--search-mode", choices=["beam", "greedy"], default="beam")
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--beam-expand", type=int, default=3)
    p.add_argument("--beam-explore-every", type=int, default=3)
    p.add_argument("--beam-exact-children", type=int, default=3)
    p.add_argument("--beam-genericity-offhop-children", type=int, default=2)
    p.add_argument("--beam-delete-probe-states", type=int, default=1)
    p.add_argument("--beam-delete-every", type=int, default=2)
    p.add_argument("--beam-delete-trigger", type=float, default=0.35)
    p.add_argument("--beam-min-soft-gain", type=float, default=0.25)
    p.add_argument("--beam-offhop-children", type=int, default=1)
    p.add_argument("--smoke", action="store_true", help="tiny one-cycle controller smoke test")
    return p.parse_args()


def main():
    args = parse_args()
    theta, rank = load_theta(args.start)
    cfg = ControllerConfig(
        goal_rank=args.goal_rank,
        seed=args.seed,
        max_cycles=args.max_cycles,
        hop_trials=args.hop_trials,
        hop_obstruction_trials=args.hop_obstruction_trials,
        hop_radius=args.hop_radius,
        offhop_trials=args.offhop_trials,
        offhop_steps_per_stage=args.offhop_steps,
        offhop_lr=args.offhop_lr,
        delete_every_hops=args.delete_every_hops,
        delete_steps_per_stage=args.delete_steps,
        delete_final_steps=args.delete_final_steps,
        coefficient_cap=args.coefficient_cap,
        susceptibility_channels=args.susceptibility_channels,
        susceptibility_steps=args.susceptibility_steps,
        susceptibility_lr=args.susceptibility_lr,
        archive_novelty_tol=args.archive_novelty_tol,
        genericity_weight=args.genericity_weight,
        genericity_deletion_weight=args.genericity_deletion_weight,
        genericity_deletion_scale=args.genericity_deletion_scale,
        beam_width=args.beam_width,
        beam_expand=args.beam_expand,
        beam_explore_every=args.beam_explore_every,
        beam_exact_children=args.beam_exact_children,
        beam_genericity_offhop_children=args.beam_genericity_offhop_children,
        beam_delete_probe_states=args.beam_delete_probe_states,
        beam_delete_every=args.beam_delete_every,
        beam_delete_trigger=args.beam_delete_trigger,
        beam_min_soft_gain=args.beam_min_soft_gain,
        beam_offhop_children=args.beam_offhop_children,
    )
    if args.smoke:
        cfg.max_cycles = 1
        cfg.continuation_max_steps = 2
        cfg.hop_trials = 1
        cfg.hop_obstruction_trials = 1
        cfg.offhop_trials = 1
        cfg.offhop_steps_per_stage = 2
        cfg.susceptibility_channels = 1
        cfg.susceptibility_steps = 2
        cfg.delete_steps_per_stage = 10
        cfg.delete_final_steps = 20
        cfg.beam_width = 2
        cfg.beam_expand = 1
        cfg.beam_explore_every = 999
        cfg.beam_exact_children = 1
        cfg.beam_genericity_offhop_children = 1
        cfg.beam_delete_probe_states = 1
        cfg.beam_offhop_children = 1
    if args.search_mode == "beam":
        q, final_rank, _ = run_beam_controller(theta, rank, cfg, args.out)
    else:
        q, final_rank, _ = run_controller(theta, rank, cfg, args.out)
    print(f"final rank={final_rank}; residual={float(residual_vector(q,mm_tensor(3),3,final_rank).norm()):.3e}")


if __name__ == "__main__":
    main()
