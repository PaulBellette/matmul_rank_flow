#!/usr/bin/env python3
"""Frozen-policy 3x3 matrix multiplication rank-23 -> rank-22 campaign.

This file is orchestration only. It deliberately does not modify the optimiser.
It imports the existing frozen specialist Pareto controller and changes only:
  * start checkpoint (rank 23),
  * goal_rank = 22,
  * stopping budget,
  * fresh RNG seed.

A numerical rank-22 landing is called a CANDIDATE, never an exact algorithm.
If one is found, it is polished with the existing tensor-only corrector and then
checked by the independent operational verifier (which imports no optimiser code).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

# Allow this orchestration file to live either in the repository root or in a
# dedicated subfolder such as rank22_campaign/.  No project code is copied.
HERE = Path(__file__).resolve().parent
for candidate_root in (HERE, HERE.parent):
    if (candidate_root / "autonomous_state_machine_3x3.py").exists():
        sys.path.insert(0, str(candidate_root))
        break

from autonomous_state_machine_3x3 import (
    ControllerConfig,
    load_theta,
    mm_tensor,
    residual_vector,
    run_beam_controller,
    save_theta,
    tensor_only_exact_polish,
    unpack,
)

DISCOVERY_SEEDS = [101, 211, 307, 401, 503]
# Fresh search randomness. These were not used in the 5-seed discovery campaign.
DEFAULT_SEARCH_SEEDS = {
    101: 1601,
    211: 1613,
    307: 1627,
    401: 1637,
    503: 1657,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def locate_start(root: Path, seed: int) -> Path:
    candidates = [
        root / "endpoint_analysis" / "polished" / f"seed_{seed}_rank23_refined.pt",
        root / f"seed_{seed}" / "controller_26_to_23" / "rank23_beam_blind.pt",
        root / f"seed_{seed}" / "controller_26_to_23" / "final.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no rank-23 start found for seed {seed}; tried:\n  " + "\n  ".join(map(str, candidates))
    )


def load_history_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))
    def finite_values(key: str):
        vals = []
        for r in rows:
            try:
                x = float(r.get(key, "nan"))
                if x == x and abs(x) != float("inf"):
                    vals.append(x)
            except Exception:
                pass
        return vals
    out = {}
    for key, name in [
        ("soft_nullity", "min_soft_nullity"),
        ("susceptibility", "min_susceptibility"),
        ("death_distance", "min_death_distance"),
        ("residual", "min_rank23_residual"),
    ]:
        vals = finite_values(key)
        out[name] = min(vals) if vals else None
    return out


def independent_verify(verifier: Path, checkpoint: Path, out_file: Path) -> tuple[bool | None, int | None]:
    if not verifier.exists():
        return None, None
    cmd = [sys.executable, str(verifier), str(checkpoint), "--trials", "5000", "--nc-trials", "500", "--tol", "1e-9"]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out_file.write_text(proc.stdout)
    return proc.returncode == 0, proc.returncode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--replication-root", type=Path, default=Path("results/replication_5seeds"))
    p.add_argument("--out", type=Path, default=Path("results/rank22_campaign"))
    p.add_argument("--max-cycles", type=int, default=120)
    p.add_argument("--seeds", type=int, nargs="*", default=DISCOVERY_SEEDS)
    p.add_argument("--independent-verifier", type=Path,
                   default=Path("independent_verification/verify_numerical_operational.py"))
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    provenance = {
        "experiment": "frozen specialist Pareto beam, rank 23 -> 22",
        "started_utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_branch": git_text(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status": git_text(["status", "--short"]),
        "python": sys.version,
        "torch": torch.__version__,
        "max_cycles": args.max_cycles,
        "launch_discovery_seeds": args.seeds,
        "search_seeds": {str(s): DEFAULT_SEARCH_SEEDS.get(s, 1000003 + s) for s in args.seeds},
        "policy_change": "none; orchestration sets goal_rank=22 and fresh RNG seed only",
        "candidate_rule": "rank 22 is numerical candidate only; independent operational PASS required before escalation",
    }
    (args.out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    try:
        (args.out / "git_diff.patch").write_text(subprocess.check_output(["git", "diff"], text=True, stderr=subprocess.DEVNULL))
    except Exception:
        pass

    results = []
    for discovery_seed in args.seeds:
        start_path = locate_start(args.replication_root, discovery_seed)
        theta, rank = load_theta(start_path)
        if rank != 23:
            raise ValueError(f"{start_path}: expected rank 23, got {rank}")
        start_res = float(residual_vector(theta, mm_tensor(3), 3, 23).norm())
        _, _, _, aa = unpack(theta, 3, 23)
        start_maxamp = float(aa.abs().max())
        if start_res > 1.0e-8:
            raise ValueError(f"{start_path}: start residual {start_res:.3e} is too large")

        search_seed = DEFAULT_SEARCH_SEEDS.get(discovery_seed, 1000003 + discovery_seed)
        run_dir = args.out / f"start_{discovery_seed}_search_{search_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "discovery_seed": discovery_seed,
            "search_seed": search_seed,
            "start": str(start_path),
            "start_sha256": sha256(start_path),
            "start_residual": start_res,
            "start_max_abs_amplitude": start_maxamp,
        }
        (run_dir / "START.json").write_text(json.dumps(manifest, indent=2))

        cfg = ControllerConfig(goal_rank=22, seed=search_seed, max_cycles=(1 if args.smoke else args.max_cycles))
        if args.smoke:
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

        print("\n" + "=" * 76, flush=True)
        print(f"START CLASS seed={discovery_seed}  search_rng={search_seed}", flush=True)
        print(f"checkpoint={start_path}", flush=True)
        print(f"residual={start_res:.3e} max|a|={start_maxamp:.3f}", flush=True)
        print("=" * 76, flush=True)

        q, final_rank, _ = run_beam_controller(theta, rank, cfg, run_dir)
        final_res = float(residual_vector(q, mm_tensor(3), 3, final_rank).norm())
        candidate = final_rank <= 22
        candidate_polished = False
        candidate_polish_res = None
        independent_pass = None
        independent_rc = None

        if candidate:
            # This does not turn the candidate into an exact certificate. It only
            # removes the controller's intentionally loose 1e-9 deletion tolerance.
            polish_cfg = replace(cfg, exact_tol=1.0e-13, beam_polish_tol=1.0e-13)
            qp, ok, rn = tensor_only_exact_polish(q, 22, polish_cfg, max_iters=40)
            candidate_polished = bool(ok)
            candidate_polish_res = float(rn)
            candidate_path = run_dir / "RANK22_CANDIDATE_POLISHED.pt"
            save_theta(candidate_path, qp, 22,
                       status="NUMERICAL_CANDIDATE_NOT_EXACT_CERTIFICATE",
                       source_start=str(start_path), search_seed=search_seed,
                       polish_residual=rn)
            independent_pass, independent_rc = independent_verify(
                args.independent_verifier, candidate_path, run_dir / "independent_operational.txt"
            )
            flag = {
                "WARNING": "NUMERICAL RANK-22 CANDIDATE ONLY; DO NOT CLAIM EXACT ALGORITHM",
                "polished": candidate_polished,
                "polish_residual": candidate_polish_res,
                "independent_operational_pass": independent_pass,
                "next_required": [
                    "independent high-precision operational check",
                    "number-field/rational exactification",
                    "exact verification of all 729 identities in independent CAS",
                ],
            }
            (run_dir / "RANK22_CANDIDATE.json").write_text(json.dumps(flag, indent=2))
            print("*** RANK-22 NUMERICAL CANDIDATE FOUND ***", flush=True)
            print(json.dumps(flag, indent=2), flush=True)

        row = {
            **manifest,
            "final_rank": final_rank,
            "final_residual": final_res,
            "rank22_candidate": candidate,
            "candidate_polished": candidate_polished,
            "candidate_polish_residual": candidate_polish_res,
            "independent_operational_pass": independent_pass,
            "independent_operational_rc": independent_rc,
            **load_history_metrics(run_dir / "beam_history.csv"),
        }
        results.append(row)
        (run_dir / "RESULT.json").write_text(json.dumps(row, indent=2))

        # Update campaign summary after every launch point.
        write_summary(args.out, results)

    write_summary(args.out, results)


def write_summary(out: Path, rows: list[dict]):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    lines = [
        "# Frozen-policy rank-23 -> rank-22 campaign",
        "",
        "> A rank-22 landing is a **numerical candidate only** until exactified and independently verified exactly.",
        "",
        "| launch | search RNG | start residual | final rank | final residual | min E_delete | min D | min soft N | candidate | independent op |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for r in rows:
        def fmt(k):
            x = r.get(k)
            return "—" if x is None else f"{x:.3e}" if isinstance(x, float) else str(x)
        lines.append(
            f"| {r['discovery_seed']} | {r['search_seed']} | {r['start_residual']:.3e} | "
            f"{r['final_rank']} | {r['final_residual']:.3e} | {fmt('min_susceptibility')} | "
            f"{fmt('min_death_distance')} | {fmt('min_soft_nullity')} | "
            f"{'YES' if r['rank22_candidate'] else 'no'} | "
            f"{('PASS' if r.get('independent_operational_pass') else 'FAIL') if r.get('independent_operational_pass') is not None else '—'} |"
        )
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
