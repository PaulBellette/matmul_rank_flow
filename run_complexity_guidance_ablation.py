#!/usr/bin/env python3
"""Matched ablation: does arithmetic-complexity guidance help rank discovery?

The rank-27 -> rank-26 collision is generated once per seed and shared across
all controller variants.  At schoolbook the eligible best collisions are
symmetry-equivalent, so there is no meaningful sparsity decision to make before
rank 26.  From that identical rank-26 checkpoint we compare:

  baseline  frozen controller, no complexity guidance
  weak      fixed weak complexity weight in beam retention/expansion policy
  delayed   complexity policy turns on only at rank <= 24
  adaptive  1/4, 1/2, 1 times the configured weight at ranks 26,25,24

No guided variant changes the local move operators, beam width, or parent-expansion
budget. Complexity is a weak policy regularizer over the same generated exact basins.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from autonomous_state_machine_3x3 import load_theta
from rank23_complexity_search import hard_support_metrics, smooth_support_objective
from geometry_flow import residual_vector
from rankflow import mm_tensor


VARIANT_TO_MODE = {
    "baseline": "off",
    "weak": "weak",
    "delayed": "delayed",
    "adaptive": "adaptive",
}
DEFAULT_SEEDS = (101, 211, 307, 401, 503)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_tee(cmd: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w") as log:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        rc = p.wait()
    return rc, time.perf_counter() - start


def parse_drop_path(text: str) -> tuple[int, str]:
    drops = [(int(a), int(b)) for a, b in re.findall(r"rank drop\s+(\d+)->(\d+)", text)]
    if not drops:
        return 0, ""
    return len(drops), " -> ".join([str(drops[0][0])] + [str(b) for _, b in drops])


def controller_result(seed: int, variant: str, ctrl_dir: Path, log: Path, runtime_s: float, rc: int,
                      tau: float, effective_tol: float) -> dict:
    text = log.read_text(errors="replace") if log.exists() else ""
    final_rank = None
    m = re.findall(r"final rank=(\d+)", text)
    if m:
        final_rank = int(m[-1])
    drops_n, drop_path = parse_drop_path(text)
    row = {
        "seed": seed,
        "variant": variant,
        "mode": VARIANT_TO_MODE[variant],
        "returncode": rc,
        "runtime_s": runtime_s,
        "final_rank": final_rank,
        "success_rank23": final_rank == 23,
        "rank_drops": drops_n,
        "drop_path": drop_path,
        "beam_generations": sum(line.startswith("beam gen=") for line in text.splitlines()),
        "final_residual": None,
        "final_additions": None,
        "final_effective_additions": None,
        "final_smooth_support": None,
    }
    final_path = ctrl_dir / "final.pt"
    if rc == 0 and final_path.exists():
        try:
            theta, rank = load_theta(final_path)
            row["final_rank"] = rank
            row["success_rank23"] = rank == 23
            row["final_residual"] = float(residual_vector(theta, mm_tensor(3), 3, rank).norm())
            row["final_additions"] = hard_support_metrics(theta, 3, rank, 1.0e-12)["hard_additions"]
            row["final_effective_additions"] = hard_support_metrics(theta, 3, rank, effective_tol)["hard_additions"]
            row["final_smooth_support"] = float(smooth_support_objective(theta, 3, rank, tau))
        except Exception as exc:
            row["final_parse_error"] = repr(exc)
    return row


def write_summary(root: Path, rows: list[dict], *, weight: float, tau: float, effective_tol: float):
    fields = [
        "seed", "variant", "mode", "returncode", "runtime_s", "final_rank", "success_rank23",
        "rank_drops", "drop_path", "beam_generations", "final_residual", "final_additions",
        "final_effective_additions", "final_smooth_support",
    ]
    with (root / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    lines = [
        "# Complexity-guided rank-discovery ablation",
        "",
        f"- smooth-support tau: `{tau:g}`",
        f"- full weak policy weight: `{weight:g}`",
        f"- effective-addition threshold: `{effective_tol:g}`",
        "- all variants share the same per-seed rank-26 collision checkpoint",
        "- no new structural zeros are snapped/frozen above rank 23",
        "",
        "| seed | variant | final rank | rank-23? | drops | path | beam gens | effective adds | structural adds | residual | runtime s |",
        "|---:|:---|---:|:---:|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        fr = r.get("final_rank") if r.get("final_rank") is not None else "—"
        adds = r.get("final_additions") if r.get("final_additions") is not None else "—"
        eadds = r.get("final_effective_additions") if r.get("final_effective_additions") is not None else "—"
        res = f"{r['final_residual']:.3e}" if r.get("final_residual") is not None else "—"
        lines.append(
            f"| {r['seed']} | {r['variant']} | {fr} | {'YES' if r.get('success_rank23') else 'no'} | "
            f"{r.get('rank_drops', 0)} | {r.get('drop_path') or '—'} | {r.get('beam_generations', 0)} | "
            f"{eadds} | {adds} | {res} | {r.get('runtime_s', 0.0):.1f} |"
        )

    lines += ["", "## Variant roll-up", "", "| variant | runs | rank-23 successes | median endpoint additions* |", "|:---|---:|---:|---:|"]
    for variant in sorted({r["variant"] for r in rows}, key=lambda v: list(VARIANT_TO_MODE).index(v)):
        rr = [r for r in rows if r["variant"] == variant]
        vals = sorted(r["final_additions"] for r in rr if r.get("success_rank23") and r.get("final_additions") is not None)
        if vals:
            n = len(vals)
            med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
            med_s = f"{med:g}"
        else:
            med_s = "—"
        lines.append(f"| {variant} | {len(rr)} | {sum(bool(r.get('success_rank23')) for r in rr)} | {med_s} |")
    lines += [
        "",
        "\\* Raw support-derived additions at the first/final rank reached by the discovery controller; no post-rank-23 cleanup or CSE is included.",
        "",
    ]
    (root / "SUMMARY.md").write_text("\n".join(lines))
    (root / "progress.json").write_text(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/complexity_guidance_ablation"))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--variants", nargs="*", choices=list(VARIANT_TO_MODE), default=list(VARIANT_TO_MODE))
    ap.add_argument("--max-cycles", type=int, default=120)
    ap.add_argument("--complexity-weight", type=float, default=0.75)
    ap.add_argument("--complexity-effective-tol", type=float, default=1.0e-3)
    ap.add_argument("--complexity-tau", type=float, default=8.0e-2)
    ap.add_argument("--delayed-rank", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = [args.seeds[0] if args.seeds else 101]
        args.variants = ["baseline", "weak"]
        args.max_cycles = 1

    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    prov = {
        "started_utc": utc_now(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "seeds": args.seeds,
        "variants": args.variants,
        "max_cycles": args.max_cycles,
        "complexity_weight": args.complexity_weight,
        "complexity_effective_tol": args.complexity_effective_tol,
        "complexity_tau": args.complexity_tau,
        "delayed_rank": args.delayed_rank,
        "note": "rank27->26 collision shared across variants because best schoolbook collision pairs are symmetry-equivalent",
    }
    try:
        prov["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        prov["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        prov["git_commit"] = "unavailable"
    (root / "PROVENANCE.json").write_text(json.dumps(prov, indent=2))

    rows: list[dict] = []
    existing = root / "progress.json"
    if existing.exists():
        try:
            rows = json.loads(existing.read_text())
        except Exception:
            rows = []
    done = {(int(r["seed"]), r["variant"]) for r in rows if int(r.get("returncode", 99)) == 0}

    for seed in args.seeds:
        seed_root = root / f"seed_{seed}"
        collision_dir = seed_root / "collision_27_to_26"
        rank26 = collision_dir / "rank26.pt"
        if not rank26.exists():
            print(f"\n=== seed {seed}: shared collision 27 -> 26 ===", flush=True)
            cmd = [sys.executable, "collision_search_3x3.py", "--mode", "demo", "--seed", str(seed), "--out", str(collision_dir)]
            rc, _ = run_tee(cmd, seed_root / "collision.log")
            if rc != 0 or not rank26.exists():
                print(f"seed {seed}: collision failed rc={rc}; variants skipped", flush=True)
                continue

        for variant in args.variants:
            if (seed, variant) in done:
                print(f"seed {seed} {variant}: already completed; skipping", flush=True)
                continue
            mode = VARIANT_TO_MODE[variant]
            ctrl_dir = seed_root / variant
            print(f"\n=== seed {seed}: {variant} ({mode}) ===", flush=True)
            cmd = [
                sys.executable, "autonomous_state_machine_3x3.py",
                "--start", str(rank26),
                "--goal-rank", "23",
                "--seed", str(seed),
                "--max-cycles", str(args.max_cycles),
                "--out", str(ctrl_dir),
                "--complexity-mode", mode,
                "--complexity-weight", str(args.complexity_weight),
                "--complexity-effective-tol", str(args.complexity_effective_tol),
                "--complexity-tau", str(args.complexity_tau),
                "--complexity-delayed-rank", str(args.delayed_rank),
            ]
            if args.smoke:
                cmd.append("--smoke")
            rc, runtime_s = run_tee(cmd, seed_root / f"{variant}.log")
            row = controller_result(seed, variant, ctrl_dir, seed_root / f"{variant}.log", runtime_s, rc, args.complexity_tau, args.complexity_effective_tol)
            rows = [r for r in rows if not (int(r["seed"]) == seed and r["variant"] == variant)]
            rows.append(row)
            rows.sort(key=lambda r: (int(r["seed"]), list(VARIANT_TO_MODE).index(r["variant"])))
            write_summary(root, rows, weight=args.complexity_weight, tau=args.complexity_tau, effective_tol=args.complexity_effective_tol)

    write_summary(root, rows, weight=args.complexity_weight, tau=args.complexity_tau, effective_tol=args.complexity_effective_tol)
    print(f"\nSummary: {root / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
