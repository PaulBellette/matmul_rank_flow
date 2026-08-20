#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def parse_final_rank(log: Path):
    if not log.exists():
        return None
    text = log.read_text(errors="replace")
    matches = re.findall(r"final rank=(\d+)", text)
    return int(matches[-1]) if matches else None


def count_rank_drops(log: Path):
    if not log.exists():
        return 0, []
    text = log.read_text(errors="replace")
    pairs = [(int(a), int(b)) for a, b in re.findall(r"rank drop\s+(\d+)->(\d+)", text)]
    return len(pairs), pairs


def parse_collision(seed_dir: Path):
    s = load_json(seed_dir / "collision_27_to_26" / "summary.json") or {}
    return {
        "collision_residual": s.get("fused_tensor_residual"),
        "collision_pair": s.get("chosen_pair_ijk") or s.get("chosen_pair"),
    }


def parse_seed(seed_dir: Path):
    seed = int(seed_dir.name.split("_", 1)[1])
    ctrl = seed_dir / "controller_26_to_23"
    beam = load_json(ctrl / "beam_summary.json") or {}
    final_rank = beam.get("final_rank")
    if final_rank is None:
        final_rank = parse_final_rank(seed_dir / "controller.log")

    drops_n, drops = count_rank_drops(seed_dir / "controller.log")
    collision = parse_collision(seed_dir)

    final_residual = None
    final_max_amp = None
    # Beam history normally contains candidate/residual information, but schema
    # has evolved. Prefer the final log line because it is stable across versions.
    log = seed_dir / "controller.log"
    if log.exists():
        text = log.read_text(errors="replace")
        m = re.findall(r"final rank=\d+; residual=([0-9.eE+-]+)", text)
        if m:
            final_residual = float(m[-1])
        m2 = re.findall(r"rank drop\s+\d+->\d+: residual=([0-9.eE+-]+) max\|a\|=([0-9.eE+-]+)", text)
        if m2 and final_rank == 23:
            final_max_amp = float(m2[-1][1])

    return {
        "seed": seed,
        "final_rank": final_rank,
        "success_rank23": final_rank == 23,
        "rank_drops": drops_n,
        "drop_path": " -> ".join([str(drops[0][0])] + [str(b) for _, b in drops]) if drops else "",
        "collision_residual": collision["collision_residual"],
        "collision_pair": json.dumps(collision["collision_pair"], separators=(",", ":")) if collision["collision_pair"] is not None else "",
        "final_residual": final_residual,
        "final_max_amp": final_max_amp,
        "beam_generations_logged": sum(1 for line in log.read_text(errors="replace").splitlines() if line.startswith("beam gen=")) if log.exists() else 0,
    }


def fmt(x, sci=False):
    if x is None or x == "":
        return "—"
    if isinstance(x, float):
        return f"{x:.3e}" if sci else f"{x:.6g}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-csv", type=Path)
    args = ap.parse_args()

    rows = [parse_seed(p) for p in sorted(args.root.glob("seed_*"), key=lambda p: int(p.name.split("_", 1)[1])) if p.is_dir()]
    successes = sum(r["success_rank23"] for r in rows)

    lines = [
        "# Five-seed end-to-end replication",
        "",
        f"- completed/visible seeds: **{len(rows)}**",
        f"- reached rank 23: **{successes}/{len(rows)}**" if rows else "- reached rank 23: **0/0**",
        "",
        "| seed | final rank | rank-23? | drops | path | collision residual | beam gens | final residual | max |a| |",
        "|---:|---:|:---:|---:|:---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {fmt(r['final_rank'])} | {'YES' if r['success_rank23'] else 'no'} | "
            f"{r['rank_drops']} | {r['drop_path'] or '—'} | {fmt(r['collision_residual'], True)} | "
            f"{r['beam_generations_logged']} | {fmt(r['final_residual'], True)} | {fmt(r['final_max_amp'])} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "This is intended as a frozen-policy replication: each seed independently chooses a symmetry-equivalent schoolbook collision and then runs the same specialist Pareto-beam controller. The only intended experimental variable is the RNG seed; `max_cycles` is a fixed stopping budget, not a tuned per-seed parameter.",
        "",
    ]

    md = "\n".join(lines)
    if args.out_md:
        args.out_md.write_text(md + "\n")
    else:
        print(md)

    if args.out_csv:
        fields = list(rows[0].keys()) if rows else ["seed", "final_rank", "success_rank23", "rank_drops", "drop_path", "collision_residual", "collision_pair", "final_residual", "final_max_amp", "beam_generations_logged"]
        with args.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
