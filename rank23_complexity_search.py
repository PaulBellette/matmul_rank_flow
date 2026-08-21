"""Search an exact rank-23 matrix-multiplication family for sparser arithmetic.

This is deliberately separate from the rank-reduction controller.  Rank is fixed
at 23 and tensor equality is treated as a hard constraint.  At an exact point we
project the gradient of a smooth support proxy into the physical tangent space,
take a finite step, and use the existing constraint corrector to land back on the
exact algorithm manifold.

The first objective is intentionally modest: reduce the support of the two input
linear-form banks and the output reconstruction bank.  It does *not* claim to
measure optimal circuit complexity; common-subexpression elimination and scalar
coefficient cost are later passes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from curvature_flow import physical_constraints, tangent_basis
from geometry_flow import pack, pinv_solve, residual_vector, unpack, unit_columns
from rankflow import mm_tensor


torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent


@dataclass
class ComplexityConfig:
    n: int = 3
    rank: int = 23
    rcond: float = 1.0e-10
    exact_tol: float = 5.0e-13
    coefficient_cap: float = 12.0

    # Smooth sparsity proxy.  The continuation schedule is intentionally broad:
    # start by redistributing medium-sized coefficients, then sharpen around
    # coefficients that are plausibly close to becoming structural zeros.
    tau_schedule: tuple[float, ...] = (0.10, 0.05, 0.02)
    generations_per_tau: int = 40
    step_size: float = 0.20
    min_step_size: float = 0.00625
    step_decay: float = 0.5
    line_search_trials: int = 6

    # Tiny beam: keep the smooth-support champion, the hard-addition champion,
    # and a couple of nearby alternatives.  Noise is tangent-space noise, so it
    # explores the exact family rather than simply perturbing tensor residual.
    beam_width: int = 4
    children_per_state: int = 3
    tangent_noise: float = 0.12
    noise_seed: int = 0

    # Diagnostics only.  hard_relative_tol is relative to the largest absolute
    # coefficient in each linear form, making the count insensitive to harmless
    # per-form rescaling.  We also log stricter/looser counts separately.
    hard_relative_tol: float = 1.0e-6
    structural_relative_tol: float = 1.0e-12
    report_tolerances: tuple[float, ...] = (1.0e-3, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-12)

    # Stop a phase after this many generations without improvement to either
    # the smooth objective or the primary hard addition count.
    patience: int = 12


@dataclass
class ComplexityMetrics:
    smooth_support: float
    structural_support: int
    structural_additions: int
    hard_support: int
    hard_additions: int
    u_support: int
    v_support: int
    output_support: int
    max_abs_amplitude: float
    residual: float


@dataclass
class SearchState:
    state_id: int
    theta: torch.Tensor
    metrics: ComplexityMetrics
    parent_id: int
    generation: int
    tau: float
    source: str
    frozen_zeros: torch.Tensor


def _normalise_rows(X: torch.Tensor) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp_min(1.0e-14)


def _effective_banks(theta: torch.Tensor, n: int, rank: int):
    """Return scale-controlled banks whose support determines raw additions.

    U and V are unit-normalised by column, exactly as reconstruction sees them.
    The output bank contains a_r W[:,r], then each *output row* is normalised so
    the smooth support proxy measures structure rather than row magnitude.
    """
    U, V, W, a = unpack(theta, n, rank)
    U = unit_columns(U)
    V = unit_columns(V)
    W = unit_columns(W)
    C = W * a.reshape(1, -1)
    Cn = _normalise_rows(C)
    return U, V, Cn, a


def smooth_support_objective(theta: torch.Tensor, n: int, rank: int, tau: float) -> torch.Tensor:
    """Differentiable log-sum sparsity objective on the three linear-form banks.

    log(1 + (x/tau)^2) is a standard smooth sparsity surrogate: under the fixed
    norm gauges it rewards concentrating mass into fewer coefficients without
    requiring a discontinuous threshold during optimisation.
    """
    U, V, C, _ = _effective_banks(theta, n, rank)
    t2 = float(tau) ** 2
    return sum(torch.log1p((X * X) / t2).sum() for X in (U, V, C))


def _relative_support_columns(X: torch.Tensor, rel_tol: float) -> torch.Tensor:
    scale = X.abs().amax(dim=0, keepdim=True).clamp_min(1.0e-300)
    return (X.abs() > rel_tol * scale).sum(dim=0)


def _relative_support_rows(X: torch.Tensor, rel_tol: float) -> torch.Tensor:
    scale = X.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-300)
    return (X.abs() > rel_tol * scale).sum(dim=1)


def hard_support_metrics(theta: torch.Tensor, n: int, rank: int, rel_tol: float) -> dict[str, int]:
    """Raw support/addition proxy before common-subexpression optimisation.

    Input forms are counted by channel (columns of U and V).  Output forms are
    counted by output entry (rows of a_r W[:,r]).  A k-term linear form costs
    k-1 additions/subtractions in the naive straight-line implementation.
    """
    U, V, W, a = unpack(theta, n, rank)
    U = unit_columns(U)
    V = unit_columns(V)
    W = unit_columns(W)
    C = W * a.reshape(1, -1)

    su = _relative_support_columns(U, rel_tol)
    sv = _relative_support_columns(V, rel_tol)
    sc = _relative_support_rows(C, rel_tol)
    additions = (
        (su - 1).clamp_min(0).sum()
        + (sv - 1).clamp_min(0).sum()
        + (sc - 1).clamp_min(0).sum()
    )
    return {
        "u_support": int(su.sum()),
        "v_support": int(sv.sum()),
        "output_support": int(sc.sum()),
        "hard_support": int(su.sum() + sv.sum() + sc.sum()),
        "hard_additions": int(additions),
    }


def complexity_metrics(theta: torch.Tensor, cfg: ComplexityConfig, tau: float) -> ComplexityMetrics:
    target = mm_tensor(cfg.n)
    residual = float(residual_vector(theta, target, cfg.n, cfg.rank).norm())
    hard = hard_support_metrics(theta, cfg.n, cfg.rank, cfg.hard_relative_tol)
    structural = hard_support_metrics(theta, cfg.n, cfg.rank, cfg.structural_relative_tol)
    _, _, _, a = unpack(theta, cfg.n, cfg.rank)
    return ComplexityMetrics(
        smooth_support=float(smooth_support_objective(theta, cfg.n, cfg.rank, tau).detach()),
        structural_support=structural["hard_support"],
        structural_additions=structural["hard_additions"],
        max_abs_amplitude=float(a.abs().max()),
        residual=residual,
        **hard,
    )


def canonical_checkpoint_to_theta(path: Path, n: int = 3) -> tuple[torch.Tensor, int]:
    """Load either a geometry checkpoint or sparse-exactification U,V,W,c file."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        rank = (obj.numel()) // (3 * n * n + 1)
        return obj.detach().clone().to(torch.float64), int(rank)
    if not isinstance(obj, dict):
        raise ValueError(f"unsupported checkpoint format: {path}")
    if "theta" in obj:
        rank = int(obj.get("rank", (obj["theta"].numel()) // (3 * n * n + 1)))
        return obj["theta"].detach().clone().to(torch.float64), rank
    if all(k in obj for k in ("U", "V", "W")) and ("c" in obj or "a" in obj):
        U = obj["U"].detach().clone().to(torch.float64)
        V = obj["V"].detach().clone().to(torch.float64)
        W = obj["W"].detach().clone().to(torch.float64)
        c = obj.get("c", obj.get("a")).detach().clone().to(torch.float64)
        rank = int(c.numel())
        # Convert the explicit multilinear gauge into geometry_flow's unit-column
        # gauge while preserving the represented tensor exactly.
        nu = U.norm(dim=0).clamp_min(1.0e-300)
        nv = V.norm(dim=0).clamp_min(1.0e-300)
        nw = W.norm(dim=0).clamp_min(1.0e-300)
        theta = pack(U / nu, V / nv, W / nw, c * nu * nv * nw)
        return theta, rank
    raise ValueError(f"unsupported checkpoint format: {path}")


def structural_zero_indices(theta: torch.Tensor, n: int, rank: int, atol: float = 1.0e-14) -> torch.Tensor:
    """Indices of factor coefficients that are already structural zeros.

    Amplitudes are deliberately excluded.  Preserving these zeros prevents the
    minimum-norm exact corrector from buying a tiny smooth-support improvement by
    densifying hundreds of coefficients that were exactly zero at the start.
    """
    factor_len = 3 * n * n * rank
    return torch.nonzero(theta[:factor_len].abs() <= atol, as_tuple=False).flatten().to(torch.long)


def threshold_zero_indices(theta: torch.Tensor, cfg: ComplexityConfig, rel_tol: float) -> torch.Tensor:
    """Factor coordinates that the raw-support metric currently regards as zero."""
    U, V, W, a = unpack(theta, cfg.n, cfg.rank)
    marks: list[int] = []
    d = cfg.n * cfg.n
    block = d * cfg.rank

    for bank_offset, X in ((0, U), (block, V)):
        scale = X.abs().amax(dim=0, keepdim=True).clamp_min(1.0e-300)
        mask = X.abs() <= rel_tol * scale
        for i, r in torch.nonzero(mask, as_tuple=False).tolist():
            marks.append(bank_offset + i * cfg.rank + r)

    # Output support is measured row-wise on C[k,r] = a_r W[k,r].  A thresholded
    # output coefficient therefore corresponds to freezing the matching W entry.
    C = unit_columns(W) * a.reshape(1, -1)
    scale = C.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-300)
    mask = C.abs() <= rel_tol * scale
    for i, r in torch.nonzero(mask, as_tuple=False).tolist():
        marks.append(2 * block + i * cfg.rank + r)

    if not marks:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(sorted(set(marks)), dtype=torch.long)


def merge_zero_indices(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0:
        return b.detach().clone().to(torch.long)
    if b.numel() == 0:
        return a.detach().clone().to(torch.long)
    return torch.unique(torch.cat([a.to(torch.long), b.to(torch.long)]), sorted=True)


def snap_threshold_zeros(
    theta: torch.Tensor,
    cfg: ComplexityConfig,
    frozen_zeros: torch.Tensor,
):
    """Try converting newly tiny coefficients into genuine structural zeros."""
    thresholded = threshold_zero_indices(theta, cfg, cfg.hard_relative_tol)
    expanded = merge_zero_indices(frozen_zeros, thresholded)
    if expanded.numel() == frozen_zeros.numel():
        return theta, frozen_zeros, True, 0.0
    x = theta.detach().clone()
    x[expanded] = 0.0
    corrected, ok, _, rn, _ = correct_sparse_constraints(x, cfg, expanded, max_iters=40)
    if not ok:
        return theta, frozen_zeros, False, rn
    _, _, _, aa = unpack(corrected, cfg.n, cfg.rank)
    if float(aa.abs().max()) > cfg.coefficient_cap:
        return theta, frozen_zeros, False, rn
    return corrected.detach(), expanded, True, rn


def sparse_constraint_vector(
    theta: torch.Tensor,
    target: torch.Tensor,
    cfg: ComplexityConfig,
    frozen_zeros: torch.Tensor,
) -> torch.Tensor:
    base = physical_constraints(theta, target, cfg.n, cfg.rank)
    if frozen_zeros.numel() == 0:
        return base
    return torch.cat([base, theta[frozen_zeros]])


def sparse_constraint_jacobian(
    theta: torch.Tensor,
    target: torch.Tensor,
    cfg: ComplexityConfig,
    frozen_zeros: torch.Tensor,
) -> torch.Tensor:
    x = theta.detach().clone().requires_grad_(True)
    return torch.autograd.functional.jacobian(
        lambda q: sparse_constraint_vector(q, target, cfg, frozen_zeros),
        x,
        vectorize=True,
    ).detach()


def correct_sparse_constraints(
    theta: torch.Tensor,
    cfg: ComplexityConfig,
    frozen_zeros: torch.Tensor,
    *,
    max_iters: int = 30,
):
    """Damped minimum-norm correction with the starting zero pattern frozen."""
    target = mm_tensor(cfg.n)
    x = theta.detach().clone()
    total_step = 0.0
    for iteration in range(max_iters + 1):
        r = sparse_constraint_vector(x, target, cfg, frozen_zeros).detach()
        rn = float(r.norm())
        if rn <= cfg.exact_tol:
            return x, True, iteration, rn, total_step
        if iteration == max_iters:
            break
        J = sparse_constraint_jacobian(x, target, cfg, frozen_zeros)
        delta = pinv_solve(J, -r, rcond=cfg.rcond)
        dn = float(delta.norm())
        if dn <= 1.0e-16:
            break
        alpha = 1.0
        accepted = False
        for _ in range(12):
            candidate = x + alpha * delta
            cr = float(sparse_constraint_vector(candidate, target, cfg, frozen_zeros).norm())
            if cr < rn:
                x = candidate
                total_step += alpha * dn
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    rn = float(sparse_constraint_vector(x, target, cfg, frozen_zeros).norm())
    return x, rn <= cfg.exact_tol, max_iters, rn, total_step


def tangent_descent_direction(
    theta: torch.Tensor, cfg: ComplexityConfig, tau: float, frozen_zeros: torch.Tensor
):
    target = mm_tensor(cfg.n)
    J = sparse_constraint_jacobian(theta, target, cfg, frozen_zeros)
    N, info = tangent_basis(J, cfg.rcond)
    if N.shape[1] == 0:
        return torch.zeros_like(theta), N, info, 0.0

    x = theta.detach().clone().requires_grad_(True)
    loss = smooth_support_objective(x, cfg.n, cfg.rank, tau)
    (g,) = torch.autograd.grad(loss, x)
    gt = N @ (N.T @ g.detach())
    gn = float(gt.norm())
    if gn <= 1.0e-14:
        return torch.zeros_like(theta), N, info, gn
    return -gt / gn, N, info, gn


def _correct_candidate(
    candidate: torch.Tensor, cfg: ComplexityConfig, frozen_zeros: torch.Tensor
):
    x, ok, iterations, rn, correction = correct_sparse_constraints(
        candidate, cfg, frozen_zeros, max_iters=30
    )
    if not ok:
        return None, iterations, rn, correction
    _, _, _, a = unpack(x, cfg.n, cfg.rank)
    if float(a.abs().max()) > cfg.coefficient_cap:
        return None, iterations, rn, correction
    return x.detach(), iterations, rn, correction


def _state_key(state: SearchState):
    m = state.metrics
    return (
        m.structural_additions, m.hard_additions, m.smooth_support,
        m.max_abs_amplitude, state.state_id
    )


def _dominates(a: SearchState, b: SearchState, eps: float = 1.0e-9) -> bool:
    ma = a.metrics
    mb = b.metrics
    va = (ma.structural_additions, ma.hard_additions, ma.smooth_support, ma.max_abs_amplitude)
    vb = (mb.structural_additions, mb.hard_additions, mb.smooth_support, mb.max_abs_amplitude)
    no_worse = all(x <= y + eps for x, y in zip(va, vb))
    strictly = any(x < y - eps for x, y in zip(va, vb))
    return no_worse and strictly


def _pareto_prune(states: list[SearchState], width: int) -> list[SearchState]:
    if len(states) <= width:
        return sorted(states, key=_state_key)
    front = [s for i, s in enumerate(states) if not any(i != j and _dominates(t, s) for j, t in enumerate(states))]
    chosen: list[SearchState] = []
    seen: set[int] = set()
    specialists = [
        min(states, key=lambda s: (s.metrics.structural_additions, s.metrics.hard_additions, s.state_id)),
        min(states, key=lambda s: (s.metrics.hard_additions, s.metrics.smooth_support, s.state_id)),
        min(states, key=lambda s: (s.metrics.smooth_support, s.metrics.hard_additions, s.state_id)),
        min(states, key=lambda s: (s.metrics.max_abs_amplitude, s.state_id)),
    ]
    for s in specialists + sorted(front, key=_state_key) + sorted(states, key=_state_key):
        if s.state_id not in seen and len(chosen) < width:
            chosen.append(s)
            seen.add(s.state_id)
    return sorted(chosen, key=_state_key)


def _save_state(path: Path, state: SearchState, cfg: ComplexityConfig):
    U, V, W, a = unpack(state.theta.detach().cpu(), cfg.n, cfg.rank)
    torch.save(
        {
            "theta": state.theta.detach().cpu(),
            "U": U,
            "V": V,
            "W": W,
            "a": a,
            "rank": cfg.rank,
            "complexity_metrics": asdict(state.metrics),
            "state_id": state.state_id,
            "parent_id": state.parent_id,
            "generation": state.generation,
            "tau": state.tau,
            "source": state.source,
            "frozen_zero_indices": state.frozen_zeros.detach().cpu(),
            "frozen_zero_count": int(state.frozen_zeros.numel()),
        },
        path,
    )


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _row(state: SearchState, cfg: ComplexityConfig, *, note: str = "", **extra):
    m = state.metrics
    row = {
        "state_id": state.state_id,
        "parent_id": state.parent_id,
        "generation": state.generation,
        "tau": state.tau,
        "source": state.source,
        "frozen_zero_count": int(state.frozen_zeros.numel()),
        "smooth_support": m.smooth_support,
        "structural_support": m.structural_support,
        "structural_additions": m.structural_additions,
        "hard_support": m.hard_support,
        "hard_additions": m.hard_additions,
        "u_support": m.u_support,
        "v_support": m.v_support,
        "output_support": m.output_support,
        "max_abs_amplitude": m.max_abs_amplitude,
        "residual": m.residual,
        "note": note,
    }
    for tol in cfg.report_tolerances:
        h = hard_support_metrics(state.theta, cfg.n, cfg.rank, tol)
        tag = f"{tol:.0e}".replace("-", "m")
        row[f"additions_rel_{tag}"] = h["hard_additions"]
        row[f"support_rel_{tag}"] = h["hard_support"]
    row.update(extra)
    return row


def run_search(start: torch.Tensor, cfg: ComplexityConfig, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    target = mm_tensor(cfg.n)
    # First make the physical unit-column gauge explicit and repair any tiny
    # numeric residual inherited from the exactification checkpoint.
    U, V, W, a = unpack(start, cfg.n, cfg.rank)
    start = pack(unit_columns(U), unit_columns(V), unit_columns(W), a)
    frozen_zeros = structural_zero_indices(start, cfg.n, cfg.rank)
    start, ok, _, rn, _ = correct_sparse_constraints(start, cfg, frozen_zeros, max_iters=30)
    if not ok:
        raise RuntimeError(f"start could not be projected to the sparse exact manifold; residual={rn:.3e}")
    print(f"preserving {int(frozen_zeros.numel())} structural factor zeros", flush=True)

    next_id = 1
    tau0 = cfg.tau_schedule[0]
    initial = SearchState(
        0, start, complexity_metrics(start, cfg, tau0), -1, 0, tau0, "initial", frozen_zeros
    )
    frontier = [initial]
    history = [_row(initial, cfg, note="initial")]
    _save_state(out / "initial.pt", initial, cfg)
    _save_state(out / "best.pt", initial, cfg)

    global_best = initial
    generation = 0
    rng = torch.Generator(device="cpu")
    rng.manual_seed(cfg.noise_seed)

    for phase_index, tau in enumerate(cfg.tau_schedule):
        phase_best_smooth = min(s.metrics.smooth_support for s in frontier)
        phase_best_adds = min(s.metrics.structural_additions for s in frontier)
        stale = 0
        step = cfg.step_size
        print(f"=== sparsity phase {phase_index}: tau={tau:g} ===", flush=True)

        # Re-score the same physical states under the new smooth scale.
        frontier = [
            SearchState(s.state_id, s.theta, complexity_metrics(s.theta, cfg, tau), s.parent_id,
                        generation, tau, f"retau:{s.source}", s.frozen_zeros)
            for s in frontier
        ]

        for local_gen in range(cfg.generations_per_tau):
            generation += 1
            children: list[SearchState] = []
            parent_rows: list[dict] = []

            for parent in frontier:
                try:
                    direction, N, info, grad_norm = tangent_descent_direction(parent.theta, cfg, tau, parent.frozen_zeros)
                except (torch._C._LinAlgError, RuntimeError) as exc:
                    history.append(_row(parent, cfg, note=f"tangent_failure:{type(exc).__name__}"))
                    continue
                if float(direction.norm()) <= 1.0e-14:
                    history.append(_row(parent, cfg, note="zero_tangent_gradient", tangent_dim=N.shape[1]))
                    continue

                # Deterministic child plus a couple of small tangent-noise blends.
                attempts = max(1, cfg.children_per_state)
                for child_index in range(attempts):
                    d = direction
                    noise_mix = 0.0
                    if child_index > 0 and N.shape[1] > 0 and cfg.tangent_noise > 0:
                        z = torch.randn(N.shape[1], generator=rng, dtype=parent.theta.dtype)
                        noise = N @ z
                        noise = noise / noise.norm().clamp_min(1.0e-14)
                        noise_mix = cfg.tangent_noise * child_index / max(1, attempts - 1)
                        d = (direction + noise_mix * noise)
                        d = d / d.norm().clamp_min(1.0e-14)

                    accepted = False
                    trial_step = step
                    for trial in range(cfg.line_search_trials):
                        candidate, corr_iters, corr_rn, corr_norm = _correct_candidate(
                            parent.theta + trial_step * d, cfg, parent.frozen_zeros
                        )
                        if candidate is not None:
                            metrics = complexity_metrics(candidate, cfg, tau)
                            child_frozen = parent.frozen_zeros
                            source = "projected_sparsity" if child_index == 0 else "projected_sparsity_noise"
                            snap_note = ""
                            # When a coefficient crosses the hard support threshold,
                            # immediately ask whether that apparent gain can be made
                            # structural.  Successful snaps become additional exact
                            # constraints for all descendants.
                            if metrics.hard_additions < parent.metrics.hard_additions:
                                snapped, snapped_frozen, snap_ok, snap_rn = snap_threshold_zeros(
                                    candidate, cfg, parent.frozen_zeros
                                )
                                if snap_ok and snapped_frozen.numel() > parent.frozen_zeros.numel():
                                    candidate = snapped
                                    child_frozen = snapped_frozen
                                    metrics = complexity_metrics(candidate, cfg, tau)
                                    source = "snap_zero"
                                    snap_note = f"snapped_{int(child_frozen.numel()-parent.frozen_zeros.numel())}_zeros"
                                elif not snap_ok:
                                    snap_note = f"snap_failed_r{snap_rn:.2e}"

                            # Keep a child when either objective makes measurable
                            # progress; Pareto pruning decides which tradeoffs survive.
                            better = (
                                metrics.smooth_support < parent.metrics.smooth_support - 1.0e-8
                                or metrics.hard_additions < parent.metrics.hard_additions
                            )
                            if better:
                                child = SearchState(
                                    next_id, candidate, metrics, parent.state_id, generation, tau,
                                    source, child_frozen,
                                )
                                next_id += 1
                                children.append(child)
                                history.append(_row(
                                    child, cfg, note="accepted_child" + (":" + snap_note if snap_note else ""),
                                    step=trial_step, tangent_dim=N.shape[1], tangent_grad_norm=grad_norm,
                                    correction_iters=corr_iters, correction_norm=corr_norm,
                                    noise_mix=noise_mix,
                                ))
                                accepted = True
                                break
                        trial_step *= cfg.step_decay
                        if trial_step < cfg.min_step_size:
                            break
                    if not accepted:
                        parent_rows.append(_row(
                            parent, cfg, note="no_improving_child", tangent_dim=N.shape[1],
                            tangent_grad_norm=grad_norm, noise_mix=noise_mix,
                        ))

            history.extend(parent_rows)
            pool = frontier + children
            frontier = _pareto_prune(pool, cfg.beam_width)

            smooth_best = min(frontier, key=lambda s: (s.metrics.smooth_support, s.metrics.hard_additions))
            add_best = min(
                frontier,
                key=lambda s: (s.metrics.structural_additions, s.metrics.hard_additions, s.metrics.smooth_support),
            )
            candidate_best = min(frontier, key=_state_key)
            if _state_key(candidate_best) < _state_key(global_best):
                global_best = candidate_best
                _save_state(out / "best.pt", global_best, cfg)

            for i, state in enumerate(frontier):
                _save_state(out / f"frontier_g{generation:03d}_{i:02d}_id{state.state_id}.pt", state, cfg)
                history.append(_row(state, cfg, note="frontier"))

            print(
                f"g={generation:03d} tau={tau:g} frontier={len(frontier)} "
                f"struct_adds={add_best.metrics.structural_additions} "
                f"adds@{cfg.hard_relative_tol:.0e}={add_best.metrics.hard_additions} "
                f"support={add_best.metrics.hard_support} "
                f"smooth={smooth_best.metrics.smooth_support:.3f} "
                f"res={candidate_best.metrics.residual:.2e} step={step:.4f}",
                flush=True,
            )
            _write_csv(out / "complexity_history.csv", history)

            improved = (
                smooth_best.metrics.smooth_support < phase_best_smooth - 1.0e-7
                or add_best.metrics.structural_additions < phase_best_adds
            )
            if improved:
                phase_best_smooth = min(phase_best_smooth, smooth_best.metrics.smooth_support)
                phase_best_adds = min(phase_best_adds, add_best.metrics.structural_additions)
                stale = 0
            else:
                stale += 1
                if stale % 4 == 0:
                    step = max(cfg.min_step_size, step * cfg.step_decay)
            if stale >= cfg.patience:
                print(f"phase tau={tau:g}: patience exhausted after {stale} stale generations", flush=True)
                break

    # Re-score final frontier under the sharpest tau for a consistent summary.
    final_tau = cfg.tau_schedule[-1]
    frontier = [
        SearchState(s.state_id, s.theta, complexity_metrics(s.theta, cfg, final_tau), s.parent_id,
                    s.generation, final_tau, s.source, s.frozen_zeros)
        for s in frontier
    ]
    best_add = min(
        frontier,
        key=lambda s: (s.metrics.structural_additions, s.metrics.hard_additions, s.metrics.smooth_support),
    )
    best_smooth = min(frontier, key=lambda s: (s.metrics.smooth_support, s.metrics.hard_additions))
    _save_state(out / "best_additions.pt", best_add, cfg)
    _save_state(out / "best_smooth.pt", best_smooth, cfg)

    summary = {
        "config": asdict(cfg),
        "initial": _row(initial, cfg, note="initial"),
        "best_additions": _row(best_add, cfg, note="best_additions"),
        "best_smooth": _row(best_smooth, cfg, note="best_smooth"),
        "final_frontier": [_row(s, cfg, note="final_frontier") for s in frontier],
        "initial_structural_factor_zeros": int(frozen_zeros.numel()),
        "best_additions_structural_factor_zeros": int(best_add.frozen_zeros.numel()),
        "interpretation": (
            "structural_additions uses a 1e-12 relative zero test while hard_additions is a 1e-6 "
            "near-zero diagnostic; both are raw support-derived counts before common-subexpression elimination. "
            "the starting structural zero pattern is frozen so the corrector cannot obtain a fake "
            "smooth improvement by densifying known zeros"
        ),
    }
    (out / "complexity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(out / "complexity_history.csv", history)
    return best_add, best_smooth, history


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/rank23_complexity"))
    p.add_argument("--tau-schedule", default="0.10,0.05,0.02")
    p.add_argument("--generations-per-tau", type=int, default=40)
    p.add_argument("--step-size", type=float, default=0.20)
    p.add_argument("--min-step-size", type=float, default=0.00625)
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--children-per-state", type=int, default=3)
    p.add_argument("--tangent-noise", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hard-relative-tol", type=float, default=1.0e-6)
    p.add_argument("--coefficient-cap", type=float, default=12.0)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    theta, rank = canonical_checkpoint_to_theta(args.checkpoint)
    if rank != 23:
        raise SystemExit(f"expected rank-23 start, got rank {rank}")
    cfg = ComplexityConfig(
        tau_schedule=_parse_float_tuple(args.tau_schedule),
        generations_per_tau=args.generations_per_tau,
        step_size=args.step_size,
        min_step_size=args.min_step_size,
        beam_width=args.beam_width,
        children_per_state=args.children_per_state,
        tangent_noise=args.tangent_noise,
        noise_seed=args.seed,
        hard_relative_tol=args.hard_relative_tol,
        coefficient_cap=args.coefficient_cap,
        patience=args.patience,
    )
    if args.smoke:
        cfg.tau_schedule = (0.05,)
        cfg.generations_per_tau = 1
        cfg.beam_width = 2
        cfg.children_per_state = 1
        cfg.line_search_trials = 2
        cfg.patience = 1
    best_add, best_smooth, _ = run_search(theta, cfg, args.out)
    print(
        f"best structural additions={best_add.metrics.structural_additions}; "
        f"near-zero additions={best_add.metrics.hard_additions}; "
        f"support={best_add.metrics.hard_support}; "
        f"residual={best_add.metrics.residual:.3e}; "
        f"best smooth={best_smooth.metrics.smooth_support:.3f}"
    )


if __name__ == "__main__":
    main()
