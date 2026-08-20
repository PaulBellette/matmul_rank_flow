"""Find a sparse, well-conditioned isotropy gauge from repeated rank-one directions.

For the tensor convention used by this repo (the third factor stores output
coefficients rather than the transposed cyclic factor), the continuous matrix-
multiplication isotropy action is

    A -> P A Q^{-1}
    B -> Q B R^{-1}
    C -> P^{-T} C R^T.

Repeated projective directions in rank-one factor matrices allow P,Q,R to be
chosen independently by small finite basis searches. This is much more stable
than unconstrained GL(3)^3 L1 optimisation.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from exactify_rank23 import canonical_channel_gauge, load_checkpoint
from geometry_flow import pack, residual_vector, unit_columns, unpack
from rankflow import mm_tensor


torch.set_default_dtype(torch.float64)


def canonical_line(v: torch.Tensor) -> torch.Tensor:
    v = v / v.norm().clamp_min(1e-15)
    k = int(torch.argmax(v.abs()))
    return -v if float(v[k]) < 0 else v


def unique_lines(vectors, tol: float = 1e-8):
    reps = []
    for v in vectors:
        v = canonical_line(v)
        if not any(abs(float(torch.dot(v, w))) > 1.0 - tol for w in reps):
            reps.append(v)
    return reps


def rank1_directions(X: torch.Tensor, ratio_tol: float = 1e-8):
    left, right, channels = [], [], []
    for r in range(X.shape[1]):
        M = X[:, r].reshape(3, 3)
        u, s, vh = torch.linalg.svd(M)
        if float(s[1] / s[0]) < ratio_tol:
            left.append(u[:, 0])
            right.append(vh[0, :])
            channels.append(r)
    return left, right, channels


def det_normalize(S: torch.Tensor):
    d = float(torch.det(S))
    S = S * abs(d) ** (-1.0 / 3.0)
    if float(torch.det(S)) < 0:
        S = S.clone()
        S[0] *= -1
    return S


def direction_score(v: torch.Tensor, softness: float = 0.03):
    z = v / v.abs().max().clamp_min(1e-15)
    a = z.abs()
    return float((a / (a + softness)).sum())


def transform_direction_score(S, primal, dual):
    Sit = torch.linalg.inv(S).T
    vals = [direction_score(S @ v) for v in primal]
    vals += [direction_score(Sit @ v) for v in dual]
    return sum(vals) / max(1, len(vals))


def choose_basis_transform(primal, dual, condition_limit: float = 20.0):
    up, ud = unique_lines(primal), unique_lines(dual)
    candidates = []
    for source, vecs in (("primal", up), ("dual", ud)):
        for ids in itertools.combinations(range(len(vecs)), 3):
            M = torch.stack([vecs[i] for i in ids], dim=1)
            if abs(float(torch.det(M))) < 1e-6:
                continue
            S = torch.linalg.inv(M) if source == "primal" else M.T
            S = det_normalize(S)
            cond = float(torch.linalg.cond(S))
            if cond > condition_limit:
                continue
            score = transform_direction_score(S, primal, dual)
            candidates.append((score, cond, source, ids, S))
    if not candidates:
        raise RuntimeError("no admissible projective basis")
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0], candidates[:10]


def apply_isotropy(theta: torch.Tensor, rank: int, P, Q, R):
    U, V, W, a = unpack(theta, 3, rank)
    U, V, W = unit_columns(U), unit_columns(V), unit_columns(W)
    A = torch.stack([U[:, r].reshape(3, 3) for r in range(rank)])
    B = torch.stack([V[:, r].reshape(3, 3) for r in range(rank)])
    C = torch.stack([W[:, r].reshape(3, 3) for r in range(rank)])

    Ap = P @ A @ torch.linalg.inv(Q)
    Bp = Q @ B @ torch.linalg.inv(R)
    Cp = torch.linalg.inv(P).T @ C @ R.T

    vecs, norms = [], []
    for X in (Ap, Bp, Cp):
        F = X.reshape(rank, 9).T.contiguous()
        n = F.norm(dim=0)
        vecs.append(F / n)
        norms.append(n)
    a2 = a * norms[0] * norms[1] * norms[2]
    return pack(vecs[0], vecs[1], vecs[2], a2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/rank23_incidence_gauge"))
    ap.add_argument("--condition-limit", type=float, default=20.0)
    ap.add_argument("--rank1-tol", type=float, default=1e-8)
    ap.add_argument("--zero-threshold", type=float, default=5e-3)
    args = ap.parse_args()

    q, rank = load_checkpoint(args.checkpoint)
    if rank != 23:
        raise SystemExit(f"expected rank 23, got {rank}")
    U, V, W, _ = unpack(q, 3, rank)
    U, V, W = unit_columns(U), unit_columns(V), unit_columns(W)
    LR = [rank1_directions(X, args.rank1_tol) for X in (U, V, W)]

    # P: A-left primal + C-left dual
    # Q: B-left primal + A-right dual
    # R: C-right primal + B-right dual
    bp, topP = choose_basis_transform(LR[0][0], LR[2][0], args.condition_limit)
    bq, topQ = choose_basis_transform(LR[1][0], LR[0][1], args.condition_limit)
    br, topR = choose_basis_transform(LR[2][1], LR[1][1], args.condition_limit)
    P, Q, R = bp[-1], bq[-1], br[-1]

    q2 = apply_isotropy(q, rank, P, Q, R)
    res = float(residual_vector(q2, mm_tensor(3), 3, rank).norm())
    Uc, Vc, Wc, c, pivots = canonical_channel_gauge(q2, 3, rank)
    allvals = torch.cat([Uc.flatten(), Vc.flatten(), Wc.flatten(), c.flatten()])
    counts = {str(t): int((allvals.abs() < t).sum()) for t in (1e-10, 1e-6, 1e-4, 1e-3, args.zero_threshold, 1e-2)}

    args.out.mkdir(parents=True, exist_ok=True)
    U2, V2, W2, a2 = unpack(q2, 3, rank)
    torch.save({
        "theta": q2, "rank": rank, "U": U2, "V": V2, "W": W2, "a": a2,
        "isotropy": {"P": P, "Q": Q, "R": R},
    }, args.out / "rank23_incidence_sparse.pt")

    def choice(x):
        return {"score": x[0], "condition": x[1], "source": x[2], "basis_ids": list(x[3])}

    report = {
        "input": str(args.checkpoint),
        "tensor_residual": res,
        "max_abs_amplitude": float(a2.abs().max()),
        "conditions": {"P": float(torch.linalg.cond(P)), "Q": float(torch.linalg.cond(Q)), "R": float(torch.linalg.cond(R))},
        "rank1_counts": {"A": len(LR[0][2]), "B": len(LR[1][2]), "C": len(LR[2][2])},
        "unique_projective_directions": {
            "A_left": len(unique_lines(LR[0][0])), "A_right": len(unique_lines(LR[0][1])),
            "B_left": len(unique_lines(LR[1][0])), "B_right": len(unique_lines(LR[1][1])),
            "C_left": len(unique_lines(LR[2][0])), "C_right": len(unique_lines(LR[2][1])),
        },
        "chosen": {"P": choice(bp), "Q": choice(bq), "R": choice(br)},
        "near_zero_counts_in_pivot_gauge": counts,
        "pivots": pivots,
    }
    (args.out / "incidence_gauge.json").write_text(json.dumps(report, indent=2) + "\n")
    md = f"""# Incidence isotropy gauge\n\n- tensor residual: `{res:.6e}`\n- condition numbers: P `{report['conditions']['P']:.3f}`, Q `{report['conditions']['Q']:.3f}`, R `{report['conditions']['R']:.3f}`\n- max |amplitude|: `{report['max_abs_amplitude']:.6f}`\n- coefficients below 1e-10 in pivot gauge: **{counts['1e-10']} / 644**\n- coefficients below {args.zero_threshold:g}: **{counts[str(args.zero_threshold)]} / 644**\n\nThe gap to the next nonzero should be checked before treating these as structural zeros.\n"""
    (args.out / "INCIDENCE_GAUGE.md").write_text(md)
    print(f"residual={res:.3e} conds=({report['conditions']['P']:.3f},{report['conditions']['Q']:.3f},{report['conditions']['R']:.3f})")
    print("near zeros:", counts)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
