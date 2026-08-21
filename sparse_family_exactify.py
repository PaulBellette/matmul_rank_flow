"""Exactify a sparse numerical rank-23 scheme along its local solution family.

Pipeline:
1. canonical per-channel pivot gauge;
2. detect a crisp structural zero pattern;
3. build the reduced Brent system with zeros and pivots fixed;
4. compute its tangent nullity;
5. greedily lock that many mobile coordinates to nearby simple rationals,
   correcting the remaining variables after every lock;
6. once isolated, refine the remaining variables at high precision;
7. recognise rationals and a common low-degree simple number field;
8. verify all 729 Brent identities exactly by arithmetic in Q[alpha]/(p(alpha)).

This is intentionally downstream from the search controller and from the
isotropy-incidence gauge. It does not use a known rank-23 endpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import scipy.linalg as la
import sympy as sp
import torch

from exactify_rank23 import (
    canonical_channel_gauge,
    exact_verify,
    expr_to_strings,
    load_checkpoint,
    numeric_from_exprs,
    tensor_from_canonical,
)
from rankflow import mm_tensor
from number_field_exact import (
    SimpleNumberField, coeffs_expr, discover_common_field, frac_str,
    verify_brent_power_basis,
)


torch.set_default_dtype(torch.float64)


@dataclass
class SparseSystem:
    base: list[torch.Tensor]
    cbase: torch.Tensor
    pivots: list[tuple[int, int, int]]
    locations: list[tuple[int, int, int]]
    x0: torch.Tensor

    def build(self, x: torch.Tensor):
        A = [z.clone() for z in self.base]
        c = self.cbase.clone()
        for k, (leg, i, r) in enumerate(self.locations):
            if leg < 3:
                A[leg][i, r] = x[k]
            else:
                c[i] = x[k]
        return A[0], A[1], A[2], c

    def residual_jacobian(self, x: torch.Tensor):
        U, V, W, c = self.build(x)
        F = (tensor_from_canonical(U, V, W, c) - mm_tensor(3)).reshape(-1)
        eye = torch.eye(9, dtype=x.dtype)
        cols = []
        for leg, i, r in self.locations:
            if leg == 0:
                D = c[r] * torch.einsum("i,j,k->ijk", eye[i], V[:, r], W[:, r])
            elif leg == 1:
                D = c[r] * torch.einsum("i,j,k->ijk", U[:, r], eye[i], W[:, r])
            elif leg == 2:
                D = c[r] * torch.einsum("i,j,k->ijk", U[:, r], V[:, r], eye[i])
            else:
                rr = i
                D = torch.einsum("i,j,k->ijk", U[:, rr], V[:, rr], W[:, rr])
            cols.append(D.reshape(-1))
        J = torch.stack(cols, dim=1)
        return F, J


def make_sparse_system(theta: torch.Tensor, rank: int, zero_threshold: float):
    U, V, W, c, pivots = canonical_channel_gauge(theta, 3, rank)
    base = [U.clone(), V.clone(), W.clone()]
    for X in base:
        X[X.abs() < zero_threshold] = 0.0
    for r, pp in enumerate(pivots):
        for leg, i in enumerate(pp):
            base[leg][i, r] = 1.0

    loc = []
    for leg, X in enumerate(base):
        for i in range(9):
            for r in range(rank):
                if i == pivots[r][leg] or float(X[i, r]) == 0.0:
                    continue
                loc.append((leg, i, r))
    # Keep all channel coefficients as variables.
    for r in range(rank):
        loc.append((3, r, 0))

    vals = []
    for leg, i, r in loc:
        vals.append(float(base[leg][i, r]) if leg < 3 else float(c[i]))
    return SparseSystem(base, c.clone(), pivots, loc, torch.tensor(vals))


def numerical_rank(J: torch.Tensor, rcond: float = 1e-10):
    s = torch.linalg.svdvals(J)
    rank = int((s > rcond * s[0]).sum()) if s.numel() else 0
    return rank, s


def svd_solve(J: torch.Tensor, b: torch.Tensor, rcond: float = 1e-10):
    U, s, Vh = torch.linalg.svd(J, full_matrices=False)
    cut = rcond * float(s[0]) if s.numel() else 0.0
    inv = torch.where(s > cut, 1.0 / s, torch.zeros_like(s))
    return Vh.T @ (inv * (U.T @ b))


def correct_with_locks(
    sys: SparseSystem,
    x: torch.Tensor,
    locked: set[int],
    targets: dict[int, float],
    *,
    tol: float = 2e-12,
    max_iters: int = 12,
    rcond: float = 1e-10,
):
    y = x.clone()
    for i, v in targets.items():
        y[i] = v
    free = [i for i in range(y.numel()) if i not in locked]

    for it in range(max_iters):
        F, J = sys.residual_jacobian(y)
        nr = float(F.norm())
        if nr < tol:
            return y, True, nr, it
        d = svd_solve(J[:, free], -F, rcond=rcond)
        accepted = False
        for ls in range(10):
            z = y.clone()
            z[free] += d * (0.5 ** ls)
            for i, v in targets.items():
                z[i] = v
            nz = float(sys.residual_jacobian(z)[0].norm())
            if nz < nr:
                y = z
                accepted = True
                break
        if not accepted:
            return y, False, nr, it
    nr = float(sys.residual_jacobian(y)[0].norm())
    return y, nr < 1e-9, nr, max_iters


def rational_candidates(x: float, mobility: float):
    out = []
    seen = set()
    for md in (4, 6, 8, 10, 12, 16, 20, 24, 32):
        f = Fraction(x).limit_denominator(md)
        if f in seen:
            continue
        seen.add(f)
        d = abs(float(f) - x)
        if d < 1e-12:
            continue
        complexity = 1.0 + 0.01 * (abs(f.numerator) + f.denominator)
        score = d / max(mobility, 1e-8) * complexity
        out.append((score, d, f))
    return out


def greedy_rational_locks(
    sys: SparseSystem,
    *,
    max_move: float = 1.0,
    rcond: float = 1e-10,
):
    x = sys.x0.clone()
    locked: set[int] = set()
    targets: dict[int, float] = {}
    target_fracs: dict[int, Fraction] = {}
    history = []

    F0, J0 = sys.residual_jacobian(x)
    rank0, _ = numerical_rank(J0, rcond)
    initial_nullity = x.numel() - rank0

    for step in range(initial_nullity + 4):
        F, J = sys.residual_jacobian(x)
        free = [i for i in range(x.numel()) if i not in locked]
        Jf = J[:, free]
        U, s, Vh = torch.linalg.svd(Jf, full_matrices=True)
        rankj = int((s > rcond * s[0]).sum()) if s.numel() else 0
        nullity = len(free) - rankj
        if nullity <= 0:
            break
        N = Vh[rankj:].T
        mobility = N.norm(dim=1)

        candidates = []
        for pos, idx in enumerate(free):
            m = float(mobility[pos])
            if m < 1e-6:
                continue
            xv = float(x[idx])
            for score, d, f in rational_candidates(xv, m):
                candidates.append((score, d, abs(f.numerator) + f.denominator, idx, f, m, xv))
        candidates.sort()

        accepted = False
        for _, d, _, idx, f, m, xv in candidates[:30]:
            ntargets = targets | {idx: float(f)}
            nlocked = locked | {idx}
            y, ok, nr, _ = correct_with_locks(sys, x, nlocked, ntargets, rcond=rcond)
            move = float((y - x).norm())
            if ok and nr < 1e-9 and move < max_move:
                history.append({
                    "step": step,
                    "index": idx,
                    "location": list(sys.locations[idx]),
                    "from": xv,
                    "target": str(f),
                    "distance": d,
                    "mobility": m,
                    "move_norm": move,
                    "residual": nr,
                })
                x = y
                locked.add(idx)
                targets[idx] = float(f)
                target_fracs[idx] = f
                accepted = True
                break
        if not accepted:
            # Singular sparse strata can expose a second-order family direction only
            # after an otherwise sensible rational lock.  In that situation the
            # remaining mobile coordinates may already be numerically zero.
            # rational_candidates() deliberately ignores <1e-12 moves, so without
            # this fallback the exactifier gives up one lock short of isolation.
            #
            # Keep the normal locking trajectory unchanged: try these exact-zero
            # locks only when no ordinary rational candidate was accepted.
            zero_fallbacks = []
            for pos, idx in enumerate(free):
                m = float(mobility[pos])
                xv = float(x[idx])
                if m < 1e-6 or xv == 0.0 or abs(xv) >= 1e-10:
                    continue
                zero_fallbacks.append((abs(xv) / max(m, 1e-8), abs(xv), idx, m, xv))
            zero_fallbacks.sort()

            for _, d, idx, m, xv in zero_fallbacks[:30]:
                f = Fraction(0)
                ntargets = targets | {idx: 0.0}
                nlocked = locked | {idx}
                y, ok, nr, _ = correct_with_locks(sys, x, nlocked, ntargets, rcond=rcond)
                move = float((y - x).norm())
                if ok and nr < 1e-9 and move < max_move:
                    history.append({
                        "step": step,
                        "index": idx,
                        "location": list(sys.locations[idx]),
                        "from": xv,
                        "target": "0",
                        "distance": d,
                        "mobility": m,
                        "move_norm": move,
                        "residual": nr,
                        "fallback": "near_zero_exact_lock",
                    })
                    x = y
                    locked.add(idx)
                    targets[idx] = 0.0
                    target_fracs[idx] = f
                    accepted = True
                    break

        if not accepted:
            break

    F, J = sys.residual_jacobian(x)
    free = [i for i in range(x.numel()) if i not in locked]
    rankf, _ = numerical_rank(J[:, free], rcond) if free else (0, torch.empty(0))
    final_nullity = len(free) - rankf
    return x, locked, targets, target_fracs, history, initial_nullity, final_nullity


def selected_independent_rows_array(J: np.ndarray, n: int, rcond: float = 1e-10):
    """Select a numerically independent equation subset by QR pivoting.

    ``J`` is equations x unknowns.  Pivoting ``J.T`` chooses equation rows.
    The returned rank is diagnostic: callers doing an exact/high-precision solve
    must not assume a float-selected square subsystem remains nonsingular.
    """
    if n == 0:
        return [], float("inf"), 0
    _, R, piv = la.qr(np.asarray(J, dtype=np.float64).T, pivoting=True, mode="economic")
    diag = np.abs(np.diag(R[:, :n]))
    if diag.size == 0:
        return [], 0.0, 0
    scale = float(diag[0])
    tol = rcond * scale if scale > 0 else 0.0
    rank = int(np.sum(diag > tol))
    rows = [int(x) for x in piv[:n]]
    return rows, float(np.min(diag)), rank


def selected_independent_rows(J: torch.Tensor, n: int, rcond: float = 1e-10):
    rows, min_diag, _ = selected_independent_rows_array(
        J.detach().cpu().numpy(), n, rcond=rcond
    )
    return rows, min_diag


def row_triplet(row: int):
    i = row // 81
    rem = row % 81
    j = rem // 9
    k = rem % 9
    return int(i), int(j), int(k)


def target_entry(i: int, j: int, k: int):
    ia, ja = divmod(i, 3)
    jb, kb = divmod(j, 3)
    ic, kc = divmod(k, 3)
    return 1 if ja == jb and ia == ic and kb == kc else 0


def high_precision_refine(
    sys: SparseSystem,
    x: torch.Tensor,
    locked: set[int],
    lock_fracs: dict[int, Fraction],
    *,
    dps: int = 130,
    rcond: float = 1e-10,
):
    """Refine the rationally locked sparse family in high precision.

    The reduced Brent system is overdetermined (729 equations).  Earlier versions
    chose one square independent subsystem in float64 and then took unrestricted
    Newton steps on that subsystem.  On ill-conditioned sparse strata a perfectly
    good local square subsystem can cease to represent the full system after a
    finite Newton step.  The result is a numerically singular square Jacobian even
    though the starting point is a high-quality solution of all 729 equations.

    We therefore globalise Newton without changing the equations:
      * repivot equation rows at every iterate from the full Jacobian;
      * compute a high-precision Newton direction on the selected square system;
      * accept it only when the norm of all 729 Brent residuals decreases;
      * if the square direction is not a descent direction, fall back to the full
        overdetermined high-precision QR least-squares direction.

    Backtracking changes only the step length, not the target equations, so an
    accepted endpoint is still refined against the exact Brent identities.
    """
    free = [i for i in range(x.numel()) if i not in locked]
    F0, J0 = sys.residual_jacobian(x)
    initial_rows, initial_min_qr_diag = selected_independent_rows(
        J0[:, free], len(free), rcond=rcond
    )

    mp.mp.dps = dps
    # Fixed sparse base: exact zeros and pivot ones; non-pivot variable locations are overwritten.
    Um = [[mp.mpf(str(float(sys.base[0][i, r]))) for r in range(23)] for i in range(9)]
    Vm = [[mp.mpf(str(float(sys.base[1][i, r]))) for r in range(23)] for i in range(9)]
    Wm = [[mp.mpf(str(float(sys.base[2][i, r]))) for r in range(23)] for i in range(9)]
    cm = [mp.mpf(str(float(sys.cbase[r]))) for r in range(23)]
    for X in (Um, Vm, Wm):
        for i in range(9):
            for r in range(23):
                # structural zeros and pivots are already exactly 0/1 in sys.base
                if X[i][r] == 0:
                    X[i][r] = mp.mpf(0)
                elif X[i][r] == 1:
                    X[i][r] = mp.mpf(1)

    for idx, fr in lock_fracs.items():
        leg, i, r = sys.locations[idx]
        v = mp.mpf(fr.numerator) / fr.denominator
        if leg == 0:
            Um[i][r] = v
        elif leg == 1:
            Vm[i][r] = v
        elif leg == 2:
            Wm[i][r] = v
        else:
            cm[i] = v

    z = mp.matrix([mp.mpf(str(float(x[idx]))) for idx in free])

    def set_unknown(zv):
        A = [row[:] for row in Um]
        B = [row[:] for row in Vm]
        C = [row[:] for row in Wm]
        cc = cm[:]
        for pos, idx in enumerate(free):
            leg, i, r = sys.locations[idx]
            v = zv[pos]
            if leg == 0:
                A[i][r] = v
            elif leg == 1:
                B[i][r] = v
            elif leg == 2:
                C[i][r] = v
            else:
                cc[i] = v
        return A, B, C, cc

    def eval_f(zv, row_ids):
        A, B, C, cc = set_unknown(zv)
        row_ids = list(row_ids)
        FF = mp.matrix(len(row_ids), 1)
        for eq, row in enumerate(row_ids):
            i, j, k = row_triplet(int(row))
            val = mp.fsum(cc[r] * A[i][r] * B[j][r] * C[k][r] for r in range(23))
            FF[eq] = val - target_entry(i, j, k)
        return FF

    def eval_fj(zv, row_ids):
        A, B, C, cc = set_unknown(zv)
        row_ids = list(row_ids)
        FF = mp.matrix(len(row_ids), 1)
        JJ = mp.matrix(len(row_ids), len(free))
        for eq, row in enumerate(row_ids):
            i, j, k = row_triplet(int(row))
            val = mp.fsum(cc[r] * A[i][r] * B[j][r] * C[k][r] for r in range(23))
            FF[eq] = val - target_entry(i, j, k)
            for pos, idx in enumerate(free):
                leg, ii, r = sys.locations[idx]
                if leg == 0:
                    JJ[eq, pos] = cc[r] * B[j][r] * C[k][r] if ii == i else 0
                elif leg == 1:
                    JJ[eq, pos] = cc[r] * A[i][r] * C[k][r] if ii == j else 0
                elif leg == 2:
                    JJ[eq, pos] = cc[r] * A[i][r] * B[j][r] if ii == k else 0
                else:
                    rr = ii
                    JJ[eq, pos] = A[i][rr] * B[j][rr] * C[k][rr]
        return FF, JJ

    def rows_from_full_jacobian(JJ_full):
        # Row selection is combinatorial only; solving remains high precision.
        Jfloat = np.array(
            [[float(JJ_full[i, j]) for j in range(len(free))]
             for i in range(729)],
            dtype=np.float64,
        )
        return selected_independent_rows_array(Jfloat, len(free), rcond=rcond)

    def extract_square(FF_full, JJ_full, row_ids):
        FF = mp.matrix([FF_full[r] for r in row_ids])
        JJ = mp.matrix(len(row_ids), len(free))
        for ii, r in enumerate(row_ids):
            for jj in range(len(free)):
                JJ[ii, jj] = JJ_full[r, jj]
        return FF, JJ

    all_rows = list(range(729))
    target_norm = mp.mpf(10) ** (-(dps - 15))
    solve_modes = []
    step_scales = []
    full_residual_history = []
    rank_history = []
    min_qr_diag = initial_min_qr_diag
    current_rows = list(initial_rows)
    repivots = 0

    # More iterations than the original pure Newton loop: backtracking may turn a
    # dangerous full step into several safe partial steps on ill-conditioned strata.
    for iteration in range(16):
        FF_full, JJ_full = eval_fj(z, all_rows)
        full_norm = mp.norm(FF_full)
        full_residual_history.append(mp.nstr(full_norm, 30))
        if full_norm < target_norm:
            break

        new_rows, new_min_diag, rank_est = rows_from_full_jacobian(JJ_full)
        rank_history.append(int(rank_est))
        if rank_est < len(free):
            raise RuntimeError(
                "high-precision Newton Jacobian is rank deficient at the current "
                f"full-system iterate: rank {rank_est} < {len(free)} free variables "
                f"at iteration {iteration}; full residual={mp.nstr(full_norm, 8)}"
            )
        if current_rows != new_rows:
            repivots += 1
        current_rows = list(new_rows)
        min_qr_diag = min(min_qr_diag, new_min_diag)
        FF, JJ = extract_square(FF_full, JJ_full, current_rows)

        directions = []
        try:
            directions.append(("square_lu", mp.lu_solve(JJ, -FF)))
        except ZeroDivisionError:
            pass

        accepted = False
        last_trial_norm = None
        # Try the cheap square Newton direction first.  It is accepted only if it
        # improves *all* Brent equations, not merely the selected square subsystem.
        for mode, dz in directions:
            alpha = mp.mpf(1)
            for _ in range(28):
                ztrial = z + alpha * dz
                trial_norm = mp.norm(eval_f(ztrial, all_rows))
                last_trial_norm = trial_norm
                if trial_norm < full_norm:
                    z = ztrial
                    solve_modes.append(mode)
                    step_scales.append(mp.nstr(alpha, 20))
                    accepted = True
                    break
                alpha /= 2
            if accepted:
                break

        if not accepted:
            # A Newton direction for one independent square subsystem need not be a
            # descent direction for the complete overdetermined system.  QR gives the
            # Gauss--Newton direction for all 729 equations without adding damping.
            try:
                dz_qr, _ = mp.qr_solve(JJ_full, -FF_full)
            except (ZeroDivisionError, ValueError) as exc:
                raise RuntimeError(
                    "full 729-equation high-precision QR solve is rank deficient "
                    f"at iteration {iteration}; full residual={mp.nstr(full_norm, 8)}"
                ) from exc

            alpha = mp.mpf(1)
            for _ in range(32):
                ztrial = z + alpha * dz_qr
                trial_norm = mp.norm(eval_f(ztrial, all_rows))
                last_trial_norm = trial_norm
                if trial_norm < full_norm:
                    z = ztrial
                    solve_modes.append("full_qr")
                    step_scales.append(mp.nstr(alpha, 20))
                    accepted = True
                    break
                alpha /= 2

        if not accepted:
            raise RuntimeError(
                "globalised high-precision Newton could not find a step decreasing "
                f"all 729 Brent residuals at iteration {iteration}; "
                f"current={mp.nstr(full_norm, 8)} "
                f"best_last_trial={mp.nstr(last_trial_norm, 8) if last_trial_norm is not None else 'n/a'}"
            )
    else:
        # The endpoint is still checked below, but make non-convergence explicit.
        final_norm = mp.norm(eval_f(z, all_rows))
        if final_norm >= target_norm:
            raise RuntimeError(
                "globalised high-precision Newton exhausted 16 iterations without "
                f"reaching the requested precision; residual={mp.nstr(final_norm, 12)}"
            )

    A, B, C, cc = set_unknown(z)
    maxerr = mp.mpf(0)
    ss = mp.mpf(0)
    for i in range(9):
        for j in range(9):
            for k in range(9):
                val = mp.fsum(cc[r] * A[i][r] * B[j][r] * C[k][r] for r in range(23))
                e = abs(val - target_entry(i, j, k))
                maxerr = max(maxerr, e)
                ss += e * e
    return {
        "free": free,
        "rows": initial_rows,
        "min_qr_diag": min_qr_diag,
        "repivots": repivots,
        "solve_modes": solve_modes,
        "step_scales": step_scales,
        "full_residual_history": full_residual_history,
        "rank_history": rank_history,
        "selected_rows": current_rows,
        "values": [z[i] for i in range(len(free))],
        "max_residual": maxerr,
        "l2_residual": mp.sqrt(ss),
    }


def recognise_high_precision(hp, *, rational_den: int = 10**6, max_field_degree: int = 12,
                             field_algdep_maxcoeff: int = 10**30,
                             field_basis_maxcoeff: int = 10**40,
                             field_individual_sample: int = 64,
                             field_pair_sample: int = 20,
                             field_triple_sample: int = 10,
                             field_aggregate_sample: int = 8):
    """Recognise refined coordinates in one simple algebraic number field.

    Unlike the original quadratic-only recogniser, this first removes rationals,
    then searches for a primitive element alpha of degree <= max_field_degree.
    Coordinates are represented exactly in the power basis 1,alpha,...,alpha^(d-1).
    Pairwise sums of low-degree coordinates are tried as primitive elements, which
    handles common cases such as biquadratic composita where no individual
    coordinate generates the full field.
    """
    mp.mp.dps = max(mp.mp.dps, 120)
    tol = mp.mpf(10) ** (-(mp.mp.dps - 30))
    rationals = {}
    unresolved = []
    hp_rows = []
    for idx, x in zip(hp["free"], hp["values"]):
        fr = Fraction(str(x)).limit_denominator(rational_den)
        err = abs(mp.mpf(fr.numerator) / fr.denominator - x)
        if err < tol:
            rationals[int(idx)] = (fr, err)
            hp_rows.append({"index": int(idx), "value": mp.nstr(x, mp.mp.dps), "rational": frac_str(fr), "rational_error": mp.nstr(err, 10)})
        else:
            unresolved.append((int(idx), x))
            hp_rows.append({"index": int(idx), "value": mp.nstr(x, mp.mp.dps), "rational": None, "rational_error": mp.nstr(err, 10)})

    if not unresolved:
        return {
            "kind": "rational",
            "degree": 1,
            "minpoly": None,
            "alpha": None,
            "representations": {idx: [fr] for idx, (fr, _err) in rationals.items()},
            "errors": {idx: err for idx, (_fr, err) in rationals.items()},
            "kinds": {idx: "rational" for idx in rationals},
            "diagnostics": {"degree_histogram": {"1": len(rationals)}, "high_precision_rows": hp_rows},
        }

    try:
        common = discover_common_field(
            unresolved,
            max_degree=max_field_degree,
            tol=tol,
            maxcoeff_algdep=field_algdep_maxcoeff,
            maxcoeff_basis=field_basis_maxcoeff,
            individual_sample=field_individual_sample,
            pair_sample=field_pair_sample,
            triple_sample=field_triple_sample,
            aggregate_sample=field_aggregate_sample,
        )
    except RuntimeError as exc:
        diagnostics = exc.args[1] if len(exc.args) > 1 and isinstance(exc.args[1], dict) else {}
        diagnostics["high_precision_rows"] = hp_rows
        raise RuntimeError("could not discover a common simple number field", diagnostics) from exc

    p = common["minpoly"]
    d = len(p) - 1
    reps = {}
    errors = {}
    kinds = {}
    for idx, (fr, err) in rationals.items():
        reps[idx] = [fr] + [Fraction(0)] * (d - 1)
        errors[idx] = err
        kinds[idx] = "rational"
    for idx, qs in common["representations"].items():
        reps[int(idx)] = list(qs)
        errors[int(idx)] = mp.mpf('0')
        kinds[int(idx)] = "algebraic" if any(qs[i] for i in range(1, len(qs))) else "rational"

    diagnostics = common["diagnostics"]
    diagnostics["high_precision_rows"] = hp_rows
    return {
        "kind": "number_field",
        "degree": d,
        "minpoly": p,
        "alpha": common["alpha"],
        "representations": reps,
        "errors": errors,
        "kinds": kinds,
        "diagnostics": diagnostics,
    }


def _field_element_json(a):
    return [frac_str(x) for x in a]


def _grid_exprs(grid):
    return [[coeffs_expr(x) for x in row] for row in grid]


def _grid_basis_json(grid):
    return [[_field_element_json(x) for x in row] for row in grid]


def build_exact_certificate(sys, locked_fracs, recognition):
    if recognition["kind"] == "rational":
        # Use a degree-one dummy field Q[alpha]/(alpha); all elements are rational.
        minpoly = [0, 1]
        degree = 1
        alpha_approx = "0"
        field_name = "Q"
    else:
        minpoly = recognition["minpoly"]
        degree = recognition["degree"]
        alpha_approx = mp.nstr(recognition["alpha"], mp.mp.dps)
        t = sp.Symbol('t')
        poly_expr = sum(sp.Integer(c) * t**i for i, c in enumerate(minpoly))
        field_name = f"Q(alpha), alpha root of {sp.sstr(poly_expr)}"

    field = SimpleNumberField(tuple(Fraction(int(c), 1) for c in minpoly))
    zero = field.zero(); one = field.one()
    U = [[zero for _ in range(23)] for _ in range(9)]
    V = [[zero for _ in range(23)] for _ in range(9)]
    W = [[zero for _ in range(23)] for _ in range(9)]
    arrs = [U, V, W]
    c = [None] * 23
    for r, pp in enumerate(sys.pivots):
        for leg, i in enumerate(pp):
            arrs[leg][i][r] = one

    recognition_rows = []
    for idx in range(len(sys.locations)):
        if idx in locked_fracs:
            fr = locked_fracs[idx]
            coeffs = [fr] + [Fraction(0)] * (degree - 1)
            kind = "rational_lock"
            err = mp.mpf(0)
        else:
            coeffs = recognition["representations"][idx]
            kind = recognition["kinds"][idx]
            err = recognition["errors"].get(idx, mp.mpf(0))
        elt = field.elt(coeffs)
        leg, i, r = sys.locations[idx]
        if leg < 3:
            arrs[leg][i][r] = elt
        else:
            c[i] = elt
        recognition_rows.append({
            "index": idx,
            "location": list(sys.locations[idx]),
            "kind": kind,
            "expr": coeffs_expr(elt),
            "power_basis": _field_element_json(elt),
            "error": mp.nstr(err, 10),
        })
    assert all(v is not None for v in c)

    ok, nonzero, failures = verify_brent_power_basis(U, V, W, c, field, 3)
    all_elts = [x for A in (U, V, W) for row in A for x in row] + c
    rational_count = sum(1 for x in all_elts if all(q == 0 for q in x[1:]))
    algebraic_count = len(all_elts) - rational_count
    return {
        "rank": 23,
        "field": field_name,
        "field_degree": degree,
        "number_field": {
            "generator": "alpha",
            "degree": degree,
            "minimal_polynomial_coefficients_ascending": [str(int(x)) for x in minpoly],
            "embedding_approx": alpha_approx,
            "selected_generator": recognition.get("diagnostics", {}).get("selected"),
        },
        "U": _grid_exprs(U),
        "V": _grid_exprs(V),
        "W": _grid_exprs(W),
        "c": [coeffs_expr(x) for x in c],
        "U_power_basis": _grid_basis_json(U),
        "V_power_basis": _grid_basis_json(V),
        "W_power_basis": _grid_basis_json(W),
        "c_power_basis": [_field_element_json(x) for x in c],
        "exact_identity": ok,
        "nonzero_exact_identities": nonzero,
        "first_failures": failures,
        "coefficient_counts": {"total": len(all_elts), "rational": rational_count, "algebraic": algebraic_count},
        "recognition": recognition_rows,
        "field_diagnostics": recognition.get("diagnostics", {}),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/rank23_sparse_exact"))
    ap.add_argument("--zero-threshold", type=float, default=5e-3)
    ap.add_argument("--dps", type=int, default=130)
    ap.add_argument("--rcond", type=float, default=1e-10)
    ap.add_argument("--max-field-degree", type=int, default=12)
    ap.add_argument("--field-algdep-maxcoeff", type=int, default=10**30)
    ap.add_argument("--field-basis-maxcoeff", type=int, default=10**40)
    ap.add_argument("--field-individual-sample", type=int, default=64)
    ap.add_argument("--field-pair-sample", type=int, default=20)
    ap.add_argument("--field-triple-sample", type=int, default=10)
    ap.add_argument("--field-aggregate-sample", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    theta, rank = load_checkpoint(args.checkpoint)
    if rank != 23:
        raise SystemExit(f"expected rank 23, got {rank}")
    Uc, Vc, Wc, cc, _ = canonical_channel_gauge(theta, 3, rank)
    vals = torch.cat([Uc.flatten(), Vc.flatten(), Wc.flatten(), cc.flatten()])
    sorted_abs = torch.sort(vals.abs()).values
    nzero = int((sorted_abs < args.zero_threshold).sum())
    next_nonzero = float(sorted_abs[nzero]) if nzero < sorted_abs.numel() else None

    sys = make_sparse_system(theta, rank, args.zero_threshold)
    initial_res = float(sys.residual_jacobian(sys.x0)[0].norm())
    F0, J0 = sys.residual_jacobian(sys.x0)
    rank0, s0 = numerical_rank(J0, args.rcond)
    nullity0 = sys.x0.numel() - rank0

    x, locked, targets, lock_fracs, hist, init_null, final_null = greedy_rational_locks(sys, rcond=args.rcond)
    locked_res = float(sys.residual_jacobian(x)[0].norm())
    family_move = float((x - sys.x0).norm())
    family_move_max = float((x - sys.x0).abs().max())

    torch.save({
        "U": sys.build(x)[0], "V": sys.build(x)[1], "W": sys.build(x)[2], "c": sys.build(x)[3],
        "pivots": sys.pivots, "locations": sys.locations, "x": x,
        "locked_indices": sorted(locked), "locked_values": {i: float(f) for i, f in lock_fracs.items()},
        "residual": locked_res,
    }, args.out / "rank23_rational_locked.pt")
    (args.out / "rational_lock_history.json").write_text(json.dumps(hist, indent=2) + "\n")

    hp = high_precision_refine(sys, x, locked, lock_fracs, dps=args.dps, rcond=args.rcond)
    # Persist the expensive high-precision solve *before* arithmetic recognition so
    # a failed recogniser is diagnosable without rerunning Newton refinement.
    hp_dump = {
        "dps": args.dps,
        "free_indices": [int(i) for i in hp["free"]],
        "values": [mp.nstr(v, args.dps) for v in hp["values"]],
        "max_residual": mp.nstr(hp["max_residual"], 30),
        "l2_residual": mp.nstr(hp["l2_residual"], 30),
        "repivots": int(hp.get("repivots", 0)),
        "solve_modes": list(hp.get("solve_modes", [])),
        "step_scales": list(hp.get("step_scales", [])),
        "full_residual_history": list(hp.get("full_residual_history", [])),
        "rank_history": [int(r) for r in hp.get("rank_history", [])],
        "selected_rows": [int(r) for r in hp.get("selected_rows", [])],
    }
    (args.out / "high_precision_values.json").write_text(json.dumps(hp_dump, indent=2) + "\n")
    try:
        rec = recognise_high_precision(
            hp, max_field_degree=args.max_field_degree,
            field_algdep_maxcoeff=args.field_algdep_maxcoeff,
            field_basis_maxcoeff=args.field_basis_maxcoeff,
            field_individual_sample=args.field_individual_sample,
            field_pair_sample=args.field_pair_sample,
            field_triple_sample=args.field_triple_sample,
            field_aggregate_sample=args.field_aggregate_sample,
        )
    except RuntimeError as exc:
        diag = exc.args[1] if len(exc.args) > 1 and isinstance(exc.args[1], dict) else {"error": str(exc)}
        (args.out / "field_recognition_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
        raise
    (args.out / "field_recognition_diagnostics.json").write_text(json.dumps(rec.get("diagnostics", {}), indent=2) + "\n")
    cert = build_exact_certificate(sys, lock_fracs, rec)
    (args.out / "rank23_exact.json").write_text(json.dumps(cert, indent=2) + "\n")

    report: dict[str, Any] = {
        "input": str(args.checkpoint),
        "structural_zero_threshold": args.zero_threshold,
        "structural_zero_count": nzero,
        "next_nonzero_abs": next_nonzero,
        "reduced_variables": int(sys.x0.numel()),
        "initial_reduced_residual": initial_res,
        "initial_jacobian_rank": rank0,
        "initial_family_nullity": nullity0,
        "rational_locks": len(locked),
        "final_family_nullity": final_null,
        "locked_residual": locked_res,
        "family_move_l2": family_move,
        "family_move_max": family_move_max,
        "high_precision_dps": args.dps,
        "high_precision_max_residual": mp.nstr(hp["max_residual"], 20),
        "high_precision_l2_residual": mp.nstr(hp["l2_residual"], 20),
        "selected_equation_min_qr_diag": hp["min_qr_diag"],
        "high_precision_repivots": int(hp.get("repivots", 0)),
        "high_precision_solve_modes": list(hp.get("solve_modes", [])),
        "field": cert["field"],
        "field_degree": cert.get("field_degree"),
        "coefficient_counts": cert["coefficient_counts"],
        "exact_identity": cert["exact_identity"],
        "nonzero_exact_identities": cert["nonzero_exact_identities"],
    }
    (args.out / "sparse_exact_report.json").write_text(json.dumps(report, indent=2) + "\n")
    md = f"""# Sparse-family exactification\n\n- structural zeros: **{nzero} / 644** at threshold `{args.zero_threshold:g}`\n- next nonzero magnitude: `{next_nonzero:.6e}`\n- reduced unknowns after zeros + pivot gauge: **{sys.x0.numel()}**\n- reduced Jacobian rank/nullity: **{rank0} / {nullity0}**\n- rational locks accepted: **{len(locked)}**\n- family move L2 / max-coordinate: `{family_move:.6e}` / `{family_move_max:.6e}`\n- high-precision full residual: `{mp.nstr(hp['l2_residual'], 8)}`\n- high-precision row repivots: **{hp.get('repivots', 0)}**\n- exact field: **{cert['field']}**\n- exact coefficient counts: **{cert['coefficient_counts']['rational']} rational + {cert['coefficient_counts']['algebraic']} algebraic = 644**\n- exact 729-identity verification: **{cert['exact_identity']}**\n\nThe rational locking step moves along the local exact solution family; it is not merely a coordinate gauge transform.\n"""
    (args.out / "SPARSE_EXACT_REPORT.md").write_text(md)

    print(f"zeros={nzero}, reduced vars={sys.x0.numel()}, family nullity={nullity0}")
    print(f"locks={len(locked)}, move={family_move:.3e}, hp_res={mp.nstr(hp['l2_residual'], 5)}, repivots={hp.get('repivots', 0)}")
    print(f"field={cert['field']} counts={cert['coefficient_counts']} exact={cert['exact_identity']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
