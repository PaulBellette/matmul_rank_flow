"""Run the first rank-23 sparsity campaign from the rational seed families."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = (211, 401)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="211,401", help="comma-separated exactified seed ids")
    p.add_argument("--out", type=Path, default=Path("runs/rank23_complexity_campaign"))
    p.add_argument("--tau-schedule", default="0.10,0.05,0.02")
    p.add_argument("--generations-per-tau", type=int, default=40)
    p.add_argument("--step-size", type=float, default=0.20)
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--children-per-state", type=int, default=3)
    p.add_argument("--tangent-noise", type=float, default=0.12)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for i, seed in enumerate(seeds):
        checkpoint = ROOT / f"results/replication_5seeds/exact_endpoints/seed_{seed}/exact/rank23_rational_locked.pt"
        if not checkpoint.exists():
            raise SystemExit(f"missing rational exactification checkpoint for seed {seed}: {checkpoint}")
        out = args.out / f"seed_{seed}"
        cmd = [
            sys.executable,
            str(ROOT / "rank23_complexity_search.py"),
            str(checkpoint),
            "--out", str(out),
            "--tau-schedule", args.tau_schedule,
            "--generations-per-tau", str(args.generations_per_tau),
            "--step-size", str(args.step_size),
            "--beam-width", str(args.beam_width),
            "--children-per-state", str(args.children_per_state),
            "--tangent-noise", str(args.tangent_noise),
            "--seed", str(1709 + 1009 * seed),
        ]
        if args.smoke:
            cmd.append("--smoke")
        print("\n" + "=" * 78, flush=True)
        print(f"START COMPLEXITY seed={seed}", flush=True)
        print(" ".join(cmd), flush=True)
        print("=" * 78, flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        summary_path = out / "complexity_summary.json"
        row = {"seed": seed, "returncode": rc, "summary": str(summary_path)}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            row["initial_structural_additions"] = summary["initial"]["structural_additions"]
            row["best_structural_additions"] = summary["best_additions"]["structural_additions"]
            row["initial_additions"] = summary["initial"]["hard_additions"]
            row["best_additions"] = summary["best_additions"]["hard_additions"]
            row["initial_support"] = summary["initial"]["hard_support"]
            row["best_support"] = summary["best_additions"]["hard_support"]
            row["best_residual"] = summary["best_additions"]["residual"]
        rows.append(row)
        (args.out / "campaign_progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        if rc != 0:
            print(f"seed {seed} failed with rc={rc}; continuing to next seed", flush=True)

    (args.out / "campaign_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("\nCampaign summary:")
    for row in rows:
        if "best_additions" in row:
            print(
                f"  seed {row['seed']}: structural additions "
                f"{row['initial_structural_additions']} -> {row['best_structural_additions']}; "
                f"near-zero additions {row['initial_additions']} -> {row['best_additions']} "
                f"support {row['initial_support']} -> {row['best_support']} "
                f"res={row['best_residual']:.3e}"
            )
        else:
            print(f"  seed {row['seed']}: FAILED rc={row['returncode']}")


if __name__ == "__main__":
    main()
