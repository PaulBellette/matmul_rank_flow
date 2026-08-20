from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch


torch.set_default_dtype(torch.float64)


def mm_tensor(n: int, device: str = "cpu") -> torch.Tensor:
    """Tensor T with T[(i,j),(j,k),(i,k)] = 1."""
    d = n * n
    T = torch.zeros((d, d, d), device=device)

    def ij(i: int, j: int) -> int:
        return i * n + j

    for i in range(n):
        for j in range(n):
            for k in range(n):
                T[ij(i, j), ij(j, k), ij(i, k)] = 1.0
    return T


def naive_factors(n: int, device: str = "cpu"):
    """The n^3-schoolbook multiplication decomposition."""
    d = n * n
    R = n**3
    U = torch.zeros((d, R), device=device)
    V = torch.zeros((d, R), device=device)
    W = torch.zeros((d, R), device=device)

    r = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j, r] = 1.0
                V[j * n + k, r] = 1.0
                W[i * n + k, r] = 1.0
                r += 1

    return U, V, W


class CPModel(torch.nn.Module):
    """
    Rank-R decomposition

        T_hat = sum_r a_r u_r ⊗ v_r ⊗ w_r

    with unit-normalized directions and bounded scalar amplitudes.
    Normalizing directions removes the simplest CP scaling gauge.
    """

    def __init__(
        self,
        n: int,
        rank: int,
        *,
        init: str,
        amplitude_max: float = 8.0,
        noise: float = 0.02,
        device: str = "cpu",
    ):
        super().__init__()
        self.n = n
        self.rank = rank
        self.amplitude_max = amplitude_max
        d = n * n

        if init == "naive":
            if rank != n**3:
                raise ValueError("naive init requires rank == n**3")
            U, V, W = naive_factors(n, device)
            U = U + noise * torch.randn_like(U)
            V = V + noise * torch.randn_like(V)
            W = W + noise * torch.randn_like(W)
            initial_a = torch.ones(rank, device=device)
        elif init == "random":
            U = torch.randn((d, rank), device=device)
            V = torch.randn((d, rank), device=device)
            W = torch.randn((d, rank), device=device)
            # raw amplitude 0 gives zero effective amplitude. A tiny random
            # value breaks symmetry without making the initial tensor huge.
            initial_a = 0.05 * torch.randn(rank, device=device)
        else:
            raise ValueError(f"unknown init {init!r}")

        self.U_raw = torch.nn.Parameter(U)
        self.V_raw = torch.nn.Parameter(V)
        self.W_raw = torch.nn.Parameter(W)

        ratio = torch.clamp(initial_a / amplitude_max, -0.999999, 0.999999)
        self.a_raw = torch.nn.Parameter(torch.atanh(ratio))

    @staticmethod
    def _unit_columns(X: torch.Tensor) -> torch.Tensor:
        return X / (X.norm(dim=0, keepdim=True) + 1.0e-12)

    def factors(self):
        U = self._unit_columns(self.U_raw)
        V = self._unit_columns(self.V_raw)
        W = self._unit_columns(self.W_raw)
        a = self.amplitude_max * torch.tanh(self.a_raw)
        return U, V, W, a

    def reconstruct(self, gates: torch.Tensor | None = None):
        U, V, W, a = self.factors()
        if gates is not None:
            a = a * gates
        T_hat = torch.einsum("ir,jr,kr,r->ijk", U, V, W, a)
        return T_hat, a


@dataclass
class Result:
    mode: str
    seed: int
    final_residual: float
    best_residual: float
    amplitudes: list[float]
    active_at_1e_3: int
    notes: str = ""


def write_rows(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def set_lr(opt, lr: float):
    for group in opt.param_groups:
        group["lr"] = lr


def fit_fixed_rank(
    *,
    n: int,
    rank: int,
    seed: int,
    steps: int,
    lr: float,
    out_dir: Path,
    device: str,
) -> Result:
    """
    Baseline: can ordinary smooth optimization fit T with the requested rank?

    This is deliberately not rank selection. It tells us whether the continuous
    parameterization itself is capable of finding a known low-rank solution.
    """
    torch.manual_seed(seed)
    T = mm_tensor(n, device)
    model = CPModel(n, rank, init="random", device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    rows = []
    best = math.inf
    best_state = None

    for step in range(steps):
        if step == steps // 3:
            set_lr(opt, lr * 0.25)
        if step == (2 * steps) // 3:
            set_lr(opt, lr * 0.05)

        T_hat, a = model.reconstruct()
        residual = ((T_hat - T) ** 2).sum()

        opt.zero_grad()
        residual.backward()
        opt.step()

        r = float(residual.detach())
        if r < best:
            best = r
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if step % max(1, steps // 200) == 0 or step == steps - 1:
            rows.append(
                {
                    "step": step,
                    "residual": r,
                    "min_abs_amplitude": float(a.detach().abs().min()),
                    "max_abs_amplitude": float(a.detach().abs().max()),
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        T_hat, a = model.reconstruct()
        final = float(((T_hat - T) ** 2).sum())
        amps = sorted(float(x) for x in a.abs().cpu())

    write_rows(out_dir / f"fit_rank{rank}_seed{seed}.csv", rows)
    torch.save(model.state_dict(), out_dir / f"fit_rank{rank}_seed{seed}.pt")

    return Result(
        mode="fit",
        seed=seed,
        final_residual=final,
        best_residual=best,
        amplitudes=amps,
        active_at_1e_3=sum(x > 1e-3 for x in amps),
    )


def soft_rank_flow(
    *,
    n: int,
    rank: int,
    seed: int,
    steps: int,
    lr: float,
    lam_max: float,
    p: float,
    out_dir: Path,
    device: str,
    init: str,
) -> Result:
    """
    Soft rank-selection experiment.

    Objective:
        beta * ||T - T_hat||_F^2
        + lambda(t) * sum_r (a_r^2 + eps(t)^2)^(p/2)

    For small p this is a smooth approximation to counting non-zero channels.
    It is intentionally non-convex.
    """
    torch.manual_seed(seed)
    T = mm_tensor(n, device)
    model = CPModel(n, rank, init=init, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    beta = 200.0
    warm = max(1, steps // 4)
    rows = []
    best = math.inf

    for step in range(steps):
        if step == steps // 2:
            set_lr(opt, lr * 0.25)
        if step == (3 * steps) // 4:
            set_lr(opt, lr * 0.05)

        frac = 0.0 if step < warm else min(1.0, (step - warm) / max(1, steps - warm - 1))
        lam = lam_max * frac
        eps = 0.10 * (1.0e-2 ** frac)

        T_hat, a = model.reconstruct()
        residual = ((T_hat - T) ** 2).sum()
        quasi_l0 = ((a * a + eps * eps) ** (p / 2.0)).sum()
        loss = beta * residual + lam * quasi_l0

        opt.zero_grad()
        loss.backward()
        opt.step()

        r = float(residual.detach())
        best = min(best, r)

        if step % max(1, steps // 200) == 0 or step == steps - 1:
            aa = a.detach().abs()
            rows.append(
                {
                    "step": step,
                    "residual": r,
                    "lambda": lam,
                    "epsilon": eps,
                    "quasi_l0": float(quasi_l0.detach()),
                    "min_abs_amplitude": float(aa.min()),
                    "max_abs_amplitude": float(aa.max()),
                    "active_at_1e_3": int((aa > 1e-3).sum()),
                }
            )

    with torch.no_grad():
        T_hat, a = model.reconstruct()
        final = float(((T_hat - T) ** 2).sum())
        amps = sorted(float(x) for x in a.abs().cpu())

    write_rows(out_dir / f"soft_rank{rank}_{init}_seed{seed}.csv", rows)
    torch.save(model.state_dict(), out_dir / f"soft_rank{rank}_{init}_seed{seed}.pt")

    return Result(
        mode="soft",
        seed=seed,
        final_residual=final,
        best_residual=best,
        amplitudes=amps,
        active_at_1e_3=sum(x > 1e-3 for x in amps),
        notes=f"init={init}, p={p}, lambda_max={lam_max}",
    )


def forced_death_flow(
    *,
    n: int,
    kill_channel: int,
    seed: int,
    steps: int,
    lr: float,
    out_dir: Path,
    device: str,
) -> Result:
    """
    Homotopy experiment from the exact schoolbook algorithm.

    One channel gets an external gate g(t): 1 -> 0.
    The remaining factors continuously re-optimise to try to keep T_hat == T.

    Bounded amplitudes prevent the killed channel from cheating by scaling
    internally like 1/g(t).
    """
    rank = n**3
    if not (0 <= kill_channel < rank):
        raise ValueError(f"kill_channel must be in [0, {rank})")

    torch.manual_seed(seed)
    T = mm_tensor(n, device)
    model = CPModel(n, rank, init="naive", device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    warm = max(1, steps // 10)
    kill_end = max(warm + 1, int(0.8 * steps))
    rows = []
    best = math.inf

    for step in range(steps):
        if step == kill_end:
            set_lr(opt, lr * 0.25)
        if step == int(0.9 * steps):
            set_lr(opt, lr * 0.05)

        if step < warm:
            gate = 1.0
        elif step < kill_end:
            gate = 1.0 - (step - warm) / (kill_end - warm)
        else:
            gate = 0.0

        gates = torch.ones(rank, device=device)
        gates[kill_channel] = gate

        T_hat, a = model.reconstruct(gates)
        residual = ((T_hat - T) ** 2).sum()

        opt.zero_grad()
        residual.backward()
        opt.step()

        r = float(residual.detach())
        best = min(best, r)

        if step % max(1, steps // 200) == 0 or step == steps - 1:
            aa = a.detach().abs()
            rows.append(
                {
                    "step": step,
                    "gate": gate,
                    "residual": r,
                    "killed_effective_amplitude": float(aa[kill_channel] * gate),
                    "min_other_abs_amplitude": float(
                        torch.cat((aa[:kill_channel], aa[kill_channel + 1 :])).min()
                    ),
                    "max_other_abs_amplitude": float(
                        torch.cat((aa[:kill_channel], aa[kill_channel + 1 :])).max()
                    ),
                }
            )

    with torch.no_grad():
        gates = torch.ones(rank, device=device)
        gates[kill_channel] = 0.0
        T_hat, a = model.reconstruct(gates)
        final = float(((T_hat - T) ** 2).sum())
        effective = a.abs().cpu()
        effective[kill_channel] = 0.0
        amps = sorted(float(x) for x in effective)

    write_rows(out_dir / f"kill_{kill_channel}_seed{seed}.csv", rows)
    torch.save(model.state_dict(), out_dir / f"kill_{kill_channel}_seed{seed}.pt")

    return Result(
        mode="kill",
        seed=seed,
        final_residual=final,
        best_residual=best,
        amplitudes=amps,
        active_at_1e_3=sum(x > 1e-3 for x in amps),
        notes=f"killed channel {kill_channel}",
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fit", "soft", "kill", "demo"], default="demo")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--rank", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--lr", type=float, default=2.0e-2)
    p.add_argument("--kill-channel", type=int, default=0)
    p.add_argument("--lambda-max", type=float, default=0.15)
    p.add_argument("--p", type=float, default=0.12)
    p.add_argument("--soft-init", choices=["random", "naive"], default="random")
    p.add_argument("--out", type=Path, default=Path("runs/rankflow"))
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []

    if args.mode in ("fit", "demo"):
        # For n=2 this is the known 7-channel target.
        target_rank = args.rank
        for seed in range(args.seed, args.seed + args.seeds):
            r = fit_fixed_rank(
                n=args.n,
                rank=target_rank,
                seed=seed,
                steps=args.steps,
                lr=args.lr,
                out_dir=args.out,
                device=args.device,
            )
            results.append(r)
            print(
                f"FIT seed={seed}: residual={r.final_residual:.3e} "
                f"best={r.best_residual:.3e}"
            )

    if args.mode in ("soft", "demo"):
        rank = args.n**3 if args.mode == "demo" else args.rank
        init = "random" if args.mode == "demo" else args.soft_init
        r = soft_rank_flow(
            n=args.n,
            rank=rank,
            seed=args.seed,
            steps=args.steps,
            lr=args.lr,
            lam_max=args.lambda_max,
            p=args.p,
            out_dir=args.out,
            device=args.device,
            init=init,
        )
        results.append(r)
        print(
            f"SOFT seed={args.seed}: residual={r.final_residual:.3e} "
            f"active={r.active_at_1e_3}/{rank}"
        )

    if args.mode in ("kill", "demo"):
        r = forced_death_flow(
            n=args.n,
            kill_channel=args.kill_channel,
            seed=args.seed,
            steps=args.steps * 3,
            lr=args.lr * 0.5,
            out_dir=args.out,
            device=args.device,
        )
        results.append(r)
        print(
            f"KILL channel={args.kill_channel}: residual={r.final_residual:.3e} "
            f"active={r.active_at_1e_3}/{args.n**3}"
        )

    summary = [asdict(r) for r in results]
    with (args.out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
