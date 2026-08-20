from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import torch
from torch.func import jacrev, vmap

from geometry_flow import pack, unpack, reconstruct, residual_vector, naive_theta, gauss_newton_correct
from rankflow import mm_tensor, naive_factors


torch.set_default_dtype(torch.float64)


def strassen_factors(device: str = "cpu"):
    """Exact classical 7-product Strassen decomposition for 2x2 matmul.

    A=[a,b;c,d], B=[e,f;g,h], output vector [C11,C12,C21,C22].
    Columns are returned with unit directions plus scalar amplitudes.
    """
    U = torch.tensor([
        [ 1, 0, 1, 0, 1,-1, 0],
        [ 0, 0, 0, 0, 1, 0, 1],
        [ 0, 1, 0, 0, 0, 1, 0],
        [ 1, 1, 0, 1, 0, 0,-1],
    ], dtype=torch.float64, device=device)
    V = torch.tensor([
        [ 1, 1, 0,-1, 0, 1, 0],
        [ 0, 0, 1, 0, 0, 1, 0],
        [ 0, 0, 0, 1, 0, 0, 1],
        [ 1, 0,-1, 0, 1, 0, 1],
    ], dtype=torch.float64, device=device)
    W = torch.tensor([
        [ 1, 0, 0, 1,-1, 0, 1],
        [ 0, 0, 1, 0, 1, 0, 0],
        [ 0, 1, 0, 1, 0, 0, 0],
        [ 1,-1, 1, 0, 0, 1, 0],
    ], dtype=torch.float64, device=device)

    nu = U.norm(dim=0)
    nv = V.norm(dim=0)
    nw = W.norm(dim=0)
    a = nu * nv * nw
    return U / nu, V / nv, W / nw, a


def strassen8_theta(device: str = "cpu") -> torch.Tensor:
    U7, V7, W7, a7 = strassen_factors(device)
    U = torch.zeros((4, 8), dtype=torch.float64, device=device)
    V = torch.zeros_like(U)
    W = torch.zeros_like(U)
    a = torch.zeros(8, dtype=torch.float64, device=device)
    U[:, :7], V[:, :7], W[:, :7], a[:7] = U7, V7, W7, a7
    # Zero channel still needs well-defined unit directions in our coordinates.
    U[0, 7] = V[0, 7] = W[0, 7] = 1.0
    return pack(U, V, W, a)


def canonicalize(theta: torch.Tensor, n: int = 2, rank: int = 8) -> torch.Tensor:
    """Fix radial CP gauge by making every factor column unit length.

    The removed scale is pushed into the scalar amplitude. This keeps the
    actual tensor represented by theta unchanged.
    """
    U, V, W, a = unpack(theta, n, rank)
    U = U.clone(); V = V.clone(); W = W.clone(); a = a.clone()
    for X in (U, V, W):
        norms = X.norm(dim=0).clamp_min(1e-14)
        a *= norms
        X /= norms
    return pack(U, V, W, a)


def term_tensor(U, V, W, a, r):
    return a[r] * torch.einsum("i,j,k->ijk", U[:, r], V[:, r], W[:, r])


def assignment_dp(cost: torch.Tensor) -> list[int]:
    """Minimum assignment; returns endpoint column chosen for each start column."""
    n = cost.shape[0]
    dp = {0: (0.0, [])}
    for i in range(n):
        ndp = {}
        for mask, (val, path) in dp.items():
            for j in range(n):
                if mask & (1 << j):
                    continue
                nm = mask | (1 << j)
                nv = val + float(cost[i, j])
                if nm not in ndp or nv < ndp[nm][0]:
                    ndp[nm] = (nv, path + [j])
        dp = ndp
    return dp[(1 << n) - 1][1]


def align_endpoint(start: torch.Tensor, end: torch.Tensor, n: int = 2, rank: int = 8):
    """Resolve CP permutation/sign gauge to make the endpoint representation nearby.

    Assignment is based on rank-one term-tensor distance. For each matched term,
    search the 8 equivalent sign gauges and choose the closest coordinates.
    The zero channel borrows the corresponding schoolbook directions.
    """
    Us, Vs, Ws, aas = unpack(canonicalize(start, n, rank), n, rank)
    Ue, Ve, We, aae = unpack(canonicalize(end, n, rank), n, rank)

    cost = torch.zeros((rank, rank), dtype=start.dtype, device=start.device)
    for i in range(rank):
        tsi = term_tensor(Us, Vs, Ws, aas, i)
        for j in range(rank):
            tej = term_tensor(Ue, Ve, We, aae, j)
            cost[i, j] = ((tsi - tej) ** 2).sum()
    assignment = assignment_dp(cost)

    U = torch.empty_like(Ue); V = torch.empty_like(Ve); W = torch.empty_like(We); a = torch.empty_like(aae)
    for i, j in enumerate(assignment):
        if abs(float(aae[j])) < 1e-14:
            U[:, i], V[:, i], W[:, i], a[i] = Us[:, i], Vs[:, i], Ws[:, i], 0.0
            continue
        best = None
        for su, sv, sw in itertools.product((-1.0, 1.0), repeat=3):
            sa = su * sv * sw  # preserve a*u*v*w
            uc = su * Ue[:, j]; vc = sv * Ve[:, j]; wc = sw * We[:, j]; ac = sa * aae[j]
            d2 = ((uc-Us[:,i])**2).sum() + ((vc-Vs[:,i])**2).sum() + ((wc-Ws[:,i])**2).sum() + (ac-aas[i])**2
            if best is None or float(d2) < best[0]:
                best = (float(d2), uc, vc, wc, ac)
        _, U[:, i], V[:, i], W[:, i], a[i] = best
    return pack(U, V, W, a), assignment


def interpolate_string(start: torch.Tensor, end: torch.Tensor, images: int) -> torch.Tensor:
    ts = torch.linspace(0.0, 1.0, images, dtype=start.dtype, device=start.device)
    pts = torch.stack([(1-t)*start + t*end for t in ts])
    return torch.stack([canonicalize(p) for p in pts])


def batch_reconstruct(points: torch.Tensor, n: int = 2, rank: int = 8):
    """Vectorized reconstruction for a whole string of parameter vectors."""
    m = points.shape[0]
    d = n * n
    block = d * rank
    U = points[:, :block].reshape(m, d, rank)
    V = points[:, block:2*block].reshape(m, d, rank)
    W = points[:, 2*block:3*block].reshape(m, d, rank)
    a = points[:, 3*block:3*block+rank]
    U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-14)
    V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-14)
    W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-14)
    return torch.einsum("mir,mjr,mkr,mr->mijk", U, V, W, a)


def batch_residual(points: torch.Tensor, target: torch.Tensor, n: int = 2, rank: int = 8):
    return (batch_reconstruct(points, n, rank) - target.unsqueeze(0)).reshape(points.shape[0], -1)


def reparameterize(points: torch.Tensor) -> torch.Tensor:
    """Linear equal-arclength redistribution in the current CP coordinate gauge."""
    seg = (points[1:] - points[:-1]).norm(dim=1)
    total = float(seg.sum())
    if total <= 1e-15:
        return points
    s = torch.cat([torch.zeros(1, dtype=points.dtype, device=points.device), torch.cumsum(seg, dim=0)])
    desired = torch.linspace(0.0, float(s[-1]), points.shape[0], dtype=points.dtype, device=points.device)
    out = [points[0]]
    j = 0
    for x in desired[1:-1]:
        while j + 1 < len(s) - 1 and x > s[j+1]:
            j += 1
        denom = (s[j+1] - s[j]).clamp_min(1e-15)
        t = (x - s[j]) / denom
        q = (1-t)*points[j] + t*points[j+1]
        out.append(canonicalize(q))
    out.append(points[-1])
    return torch.stack(out)


def string_step(points: torch.Tensor, target: torch.Tensor, lr: float):
    """Vectorized zero-temperature string step: descend potential normal to path."""
    x = points.detach().clone().requires_grad_(True)
    rs = batch_residual(x, target)
    V = 0.5 * (rs * rs).sum(dim=1)
    (g,) = torch.autograd.grad(V.sum(), x)

    tau = points[2:] - points[:-2]
    tau = tau / tau.norm(dim=1, keepdim=True).clamp_min(1e-14)
    gi = g[1:-1]
    g_perp = gi - (gi * tau).sum(dim=1, keepdim=True) * tau
    gn = g_perp.norm(dim=1, keepdim=True)
    scale = torch.clamp(5.0 / gn.clamp_min(1e-15), max=1.0)

    new = points.detach().clone()
    updated = points[1:-1] - lr * scale * g_perp
    # Canonicalization is cheap at this scale and keeps radial CP gauge fixed.
    new[1:-1] = torch.stack([canonicalize(q) for q in updated])
    return new, [float(v) for v in V[1:-1].detach()], [float(v) for v in gn[:,0].detach()]


def whitened_string_step(points: torch.Tensor, target: torch.Tensor, alphas=None):
    """Jacobian-whitened normal step with a barrier line search.

    Gradient flow near F=0 evolves under J^T J and can be extremely stiff.
    Here we solve J delta ~= -F by a batched truncated SVD, remove motion along
    the current string tangent, then line-search after equal-arclength
    reparameterization.  This is the direct analogue of changing to better
    linearized coordinates rather than merely increasing the learning rate.
    """
    if alphas is None:
        alphas = (0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005)

    f = lambda q: residual_vector(q, target, 2, 8)
    interior = points[1:-1].detach()
    r = vmap(f)(interior)
    J = vmap(jacrev(f))(interior)
    U, S, Vh = robust_svd(J, full_matrices=False)
    invS = torch.where(S > 1e-10 * S[:, :1], 1.0 / S, torch.zeros_like(S))
    coeff = torch.matmul(U.transpose(-2, -1), (-r).unsqueeze(-1)).squeeze(-1) * invS
    delta = torch.matmul(Vh.transpose(-2, -1), coeff.unsqueeze(-1)).squeeze(-1)

    tau = points[2:] - points[:-2]
    tau = tau / tau.norm(dim=1, keepdim=True).clamp_min(1e-14)
    delta = delta - (delta * tau).sum(dim=1, keepdim=True) * tau

    old = path_diagnostics(points, target)
    best_points = points
    best_diag = old
    best_alpha = None
    for alpha in alphas:
        candidate = points.clone()
        candidate[1:-1] = torch.stack([canonicalize(q) for q in (interior + alpha * delta)])
        candidate = reparameterize(candidate)
        diag = path_diagnostics(candidate, target)
        if diag["max_residual"] < best_diag["max_residual"]:
            best_points, best_diag, best_alpha = candidate, diag, alpha
    return best_points, best_diag, best_alpha


def path_diagnostics(points: torch.Tensor, target: torch.Tensor):
    with torch.no_grad():
        rs = batch_residual(points, target)
        rn = rs.norm(dim=1)
        seg = (points[1:] - points[:-1]).norm(dim=1)
        return {
            "max_residual": float(rn.max()),
            "mean_residual": float(rn.mean()),
            "barrier_index": int(torch.argmax(rn)),
            "path_length": float(seg.sum()),
            "min_segment": float(seg.min()),
            "max_segment": float(seg.max()),
            "residuals": [float(x) for x in rn],
        }


def run_string(*, images: int, steps: int, lr: float, reparam_every: int, out: Path, device: str, resume: Path | None = None, whiten_every: int = 0):
    target = mm_tensor(2, device)
    start = canonicalize(naive_theta(2, device))
    raw_end = canonicalize(strassen8_theta(device))
    end, assignment = align_endpoint(start, raw_end)

    r_start = float(residual_vector(start, target, 2, 8).norm())
    r_end = float(residual_vector(end, target, 2, 8).norm())
    print(f"endpoint residuals: schoolbook={r_start:.3e}, strassen+0={r_end:.3e}")
    print("endpoint assignment:", assignment)

    if resume is not None:
        points = torch.load(resume, map_location=device).to(dtype=torch.float64, device=device)
        images = points.shape[0]
        points[0] = start
        points[-1] = end
    else:
        points = interpolate_string(start, end, images)
    init = path_diagnostics(points, target)
    print(f"initial string: max_res={init['max_residual']:.6g}, length={init['path_length']:.4g}")

    history = []
    best = init["max_residual"]
    best_points = points.clone()
    current_lr = lr

    for step in range(steps + 1):
        if step > 0:
            if whiten_every > 0 and step % whiten_every == 0:
                proposal, dnew, alpha = whitened_string_step(points, target)
                if alpha is not None:
                    points = proposal
            else:
                proposal, _, _ = string_step(points, target, current_lr)
                if step % reparam_every == 0:
                    proposal = reparameterize(proposal)
                dnew = path_diagnostics(proposal, target)
                dold = path_diagnostics(points, target)
                # Conservative global line search if the barrier jumps badly.
                if dnew["max_residual"] > max(dold["max_residual"] * 1.20, dold["max_residual"] + 1e-8):
                    current_lr *= 0.5
                else:
                    points = proposal
            dcur = path_diagnostics(points, target)
            if dcur["max_residual"] < best:
                best = dcur["max_residual"]
                best_points = points.clone()

        if step % max(1, steps // 100) == 0 or step == steps:
            d = path_diagnostics(points, target)
            row = {"step": step, "lr": current_lr, **{k:v for k,v in d.items() if k != "residuals"}}
            history.append(row)
            print(f"step {step:5d}: max={d['max_residual']:.4e} mean={d['mean_residual']:.4e} idx={d['barrier_index']:2d} L={d['path_length']:.4f} lr={current_lr:.2e}")

    points = best_points
    final = path_diagnostics(points, target)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(points.cpu(), out / "string_points.pt")
    with (out / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
    rows = []
    for i, r in enumerate(final["residuals"]):
        _, _, _, a = unpack(points[i], 2, 8)
        rows.append({"image":i, "s":i/(images-1), "residual":r, "min_abs_amplitude":float(a.abs().min()), "max_abs_amplitude":float(a.abs().max())})
    with (out / "final_path.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary = {"images":images,"steps":steps,"initial":init,"final":final,"best_max_residual":best,"endpoint_assignment":assignment}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("final best max residual:", best)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",type=int,default=31)
    p.add_argument("--steps",type=int,default=3000)
    p.add_argument("--lr",type=float,default=0.05)
    p.add_argument("--reparam-every",type=int,default=5)
    p.add_argument("--out",type=Path,default=Path("runs/string"))
    p.add_argument("--device",default="cpu")
    p.add_argument("--resume",type=Path,default=None,help="resume from a saved string_points.pt")
    p.add_argument("--whiten-every",type=int,default=0,help="attempt a Jacobian-whitened normal step every N iterations")
    args=p.parse_args()
    run_string(images=args.images,steps=args.steps,lr=args.lr,reparam_every=args.reparam_every,out=args.out,device=args.device,resume=args.resume,whiten_every=args.whiten_every)

if __name__ == "__main__":
    main()
