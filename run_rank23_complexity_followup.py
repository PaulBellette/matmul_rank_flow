"""Exactify low-support rank-23 campaign winners and compare linear circuit cost."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from exactify_rank23 import canonical_channel_gauge, load_checkpoint
from rank23_linear_circuit import analyze_certificate

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, default=Path("runs/rank23_complexity_campaign"))
    p.add_argument("--out", type=Path, default=Path("runs/rank23_complexity_followup"))
    p.add_argument("--seeds", default="211,401")
    p.add_argument("--dps", type=int, default=130)
    p.add_argument("--rcond", type=float, default=1e-10)
    p.add_argument("--max-field-degree", type=int, default=12)
    p.add_argument("--skip-exactify", action="store_true", help="only analyze existing exact outputs")
    return p.parse_args()


def _factor_location(index: int, rank: int):
    block = 9 * rank
    if index < 0 or index >= 3 * block:
        raise ValueError(f"frozen index {index} is not a factor coordinate")
    leg = index // block
    rem = index % block
    i, r = divmod(rem, rank)
    return leg, i, r


def prepare_winner(checkpoint: Path, out: Path):
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "theta" not in obj:
        raise ValueError(f"complexity winner must be a saved search-state dict: {checkpoint}")
    theta, rank = load_checkpoint(checkpoint)
    if rank != 23:
        raise ValueError(f"expected rank 23, got {rank}")
    frozen = obj.get("frozen_zero_indices")
    if frozen is None:
        raise ValueError("winner checkpoint lacks frozen_zero_indices")
    frozen = frozen.to(torch.long).flatten()
    x = theta.detach().clone()
    x[frozen] = 0.0

    # Canonical channel gauge only rescales columns, so zero locations are preserved.
    U, V, W, c, _ = canonical_channel_gauge(x, 3, rank)
    banks = (U, V, W)
    frozen_locs = {_factor_location(int(k), rank) for k in frozen.tolist()}
    nonfrozen_abs = []
    forced_abs = []
    for leg, X in enumerate(banks):
        for i in range(9):
            for r in range(rank):
                v = abs(float(X[i, r]))
                if (leg, i, r) in frozen_locs:
                    forced_abs.append(v)
                elif v:
                    nonfrozen_abs.append(v)
    if not nonfrozen_abs:
        raise ValueError("no non-frozen factor coefficients")
    min_nonfrozen = min(nonfrozen_abs)
    max_forced = max(forced_abs, default=0.0)
    # We have explicitly set the frozen coordinates to zero. Pick a threshold far
    # below every remaining nonzero so sparse_family_exactify cannot invent zeros.
    threshold = min(1e-10, min_nonfrozen * 0.1)
    if max_forced >= threshold:
        raise RuntimeError(
            f"cannot separate forced zeros ({max_forced:.3e}) from nonzeros "
            f"({min_nonfrozen:.3e})"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "theta": x,
        "rank": rank,
        "source": str(checkpoint),
        "frozen_zero_indices": frozen,
        "frozen_zero_count": int(frozen.numel()),
        "safe_zero_threshold": threshold,
        "min_nonfrozen_canonical_abs": min_nonfrozen,
    }, out)
    return {
        "prepared_checkpoint": str(out),
        "frozen_zero_count": int(frozen.numel()),
        "safe_zero_threshold": threshold,
        "min_nonfrozen_canonical_abs": min_nonfrozen,
    }


def verify_mask_survives(exact_json: Path, prepared_pt: Path):
    # Exact certificate is in canonical gauge; map the original frozen locations
    # directly to U/V/W entries and require literal exact zero strings/power-basis zeros.
    cert = json.loads(exact_json.read_text())
    prep = torch.load(prepared_pt, map_location="cpu", weights_only=False)
    frozen = prep["frozen_zero_indices"].to(torch.long).tolist()
    rank = int(prep["rank"])
    names = ("U_power_basis", "V_power_basis", "W_power_basis")
    bad = []
    for idx in frozen:
        leg, i, r = _factor_location(int(idx), rank)
        coeffs = cert[names[leg]][i][r]
        if any(str(q) not in ("0", "0/1") for q in coeffs):
            bad.append({"flat_index": int(idx), "leg": leg, "i": i, "r": r, "coeffs": coeffs})
    return {"checked": len(frozen), "all_exact_zero": not bad, "failures": bad[:20]}


def markdown_report(rows):
    lines = [
        "# Rank-23 complexity follow-up",
        "",
        "Low-support campaign winners are exactified independently and their linear-form",
        "addition counts are compared before and after a deterministic greedy exact CSE pass.",
        "CSE counts are heuristic straight-line-program upper bounds, not global minima; scalar",
        "constant multiplications are not charged in the addition count.",
        "",
        "| seed | winner exact | snapped zeros exact | naive adds baseline -> numeric -> exact | greedy-CSE adds old -> exact | field |",
        "|---:|:---:|:---:|---:|---:|:---|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r['seed']} | no | — | — | — | {r.get('error','failed')} |")
            continue
        numeric = r.get("campaign_best_structural_additions")
        numeric_text = str(numeric) if numeric is not None else "?"
        lines.append(
            f"| {r['seed']} | yes | {'yes' if r['zero_mask']['all_exact_zero'] else 'NO'} | "
            f"{r['baseline']['naive_additions']} -> {numeric_text} -> {r['winner']['naive_additions']} | "
            f"{r['baseline']['greedy_cse_additions']} -> {r['winner']['greedy_cse_additions']} | "
            f"{r['winner']['field']} |"
        )
    lines += [
        "",
        "The exact-zero check is the key guardrail: every coefficient deliberately snapped by",
        "the complexity search must remain zero in the symbolic certificate.",
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for seed in seeds:
        seed_out = args.out / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        winner_pt = args.campaign / f"seed_{seed}" / "best_additions.pt"
        baseline_json = ROOT / f"results/replication_5seeds/exact_endpoints/seed_{seed}/exact/rank23_exact.json"
        prepared = seed_out / "winner_for_exactify.pt"
        exact_dir = seed_out / "exact"
        exact_json = exact_dir / "rank23_exact.json"
        row = {"seed": seed, "winner_checkpoint": str(winner_pt)}
        campaign_summary = args.campaign / f"seed_{seed}" / "complexity_summary.json"
        if campaign_summary.exists():
            try:
                cs = json.loads(campaign_summary.read_text())
                numeric_best = cs.get("best_structural_additions")
                if numeric_best is None:
                    numeric_best = cs.get("best_additions")
                    if isinstance(numeric_best, dict):
                        numeric_best = numeric_best.get("structural_additions", numeric_best.get("hard_additions"))
                row["campaign_best_structural_additions"] = numeric_best
            except Exception:
                pass
        try:
            if not winner_pt.exists():
                raise FileNotFoundError(f"missing campaign winner {winner_pt}")
            if not baseline_json.exists():
                raise FileNotFoundError(f"missing baseline exact certificate {baseline_json}")
            prep = prepare_winner(winner_pt, prepared)
            row["preparation"] = prep
            if not args.skip_exactify:
                cmd = [
                    sys.executable, str(ROOT / "sparse_family_exactify.py"), str(prepared),
                    "--out", str(exact_dir),
                    "--zero-threshold", f"{prep['safe_zero_threshold']:.17g}",
                    "--dps", str(args.dps), "--rcond", str(args.rcond),
                    "--max-field-degree", str(args.max_field_degree),
                ]
                print("\n" + "=" * 78, flush=True)
                print(f"EXACTIFY COMPLEXITY WINNER seed={seed}", flush=True)
                print(" ".join(cmd), flush=True)
                print("=" * 78, flush=True)
                rc = subprocess.run(cmd, cwd=ROOT).returncode
                if rc:
                    raise RuntimeError(f"exactifier failed rc={rc}; inspect {exact_dir}")
            if not exact_json.exists():
                raise FileNotFoundError(f"missing exact output {exact_json}")
            cert = json.loads(exact_json.read_text())
            if not cert.get("exact_identity"):
                raise RuntimeError("winner certificate does not verify all 729 Brent identities")
            independent_cmd = [
                sys.executable, str(ROOT / "independent_verification/verify_sympy_exact.py"),
                str(exact_json), "--trials", "20", "--nc-trials", "5",
            ]
            independent = subprocess.run(independent_cmd, cwd=ROOT, capture_output=True, text=True)
            (seed_out / "independent_verify.log").write_text(independent.stdout + independent.stderr)
            if independent.returncode:
                raise RuntimeError("independent SymPy operational verifier failed")
            row["independent_verifier"] = "pass"
            zero_mask = verify_mask_survives(exact_json, prepared)
            if not zero_mask["all_exact_zero"]:
                raise RuntimeError(f"{len(zero_mask['failures'])} snapped zeros did not survive exactification")
            baseline = analyze_certificate(baseline_json)
            winner = analyze_certificate(exact_json)
            row.update({"status": "ok", "zero_mask": zero_mask, "baseline": baseline, "winner": winner})
        except Exception as exc:
            row.update({"status": "failed", "error": str(exc)})
        rows.append(row)
        (args.out / "followup_progress.json").write_text(json.dumps(rows, indent=2) + "\n")

    (args.out / "followup_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.out / "COMPLEXITY_FOLLOWUP.md").write_text(markdown_report(rows) + "\n")
    print("\n" + markdown_report(rows))
    if any(r.get("status") != "ok" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
