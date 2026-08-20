#!/usr/bin/env bash
set -uo pipefail

# Five-seed end-to-end replication of the autonomous 3x3 rank-flow search.
# Stage 1: schoolbook rank 27 -> rank 26 using collision_search_3x3.py.
# Stage 2: rank 26 -> goal rank 23 using autonomous_state_machine_3x3.py.
#
# Usage:
#   bash run_5seed_replication.sh
#   bash run_5seed_replication.sh results/replication_5seeds 120 0 1 2 3 4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARY_SCRIPT="$SCRIPT_DIR/summarize_5seed_replication.py"

OUT_ROOT="${1:-results/replication_5seeds}"
MAX_CYCLES="${2:-120}"
shift $(( $# >= 1 ? 1 : 0 )) || true
shift $(( $# >= 1 ? 1 : 0 )) || true

if (( $# > 0 )); then
  SEEDS=("$@")
else
  SEEDS=(101 211 307 401 503)
fi

mkdir -p "$OUT_ROOT"

# Freeze provenance at launch without modifying the repo.
{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$(python3 --version 2>&1)"
  echo "max_cycles=$MAX_CYCLES"
  echo "seeds=${SEEDS[*]}"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_branch=$(git rev-parse --abbrev-ref HEAD)"
    echo "git_dirty=$(test -n \"$(git status --porcelain)\" && echo yes || echo no)"
  else
    echo "git_commit=not_a_git_repo"
  fi
} > "$OUT_ROOT/PROVENANCE.txt"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git status --short > "$OUT_ROOT/git_status.txt" 2>&1 || true
  git diff > "$OUT_ROOT/git_diff.patch" 2>&1 || true
fi

printf 'seed\tcollision_rc\tcontroller_rc\tstarted_utc\tfinished_utc\n' > "$OUT_ROOT/run_status.tsv"

for seed in "${SEEDS[@]}"; do
  SEED_DIR="$OUT_ROOT/seed_${seed}"
  COLL_DIR="$SEED_DIR/collision_27_to_26"
  CTRL_DIR="$SEED_DIR/controller_26_to_23"
  mkdir -p "$COLL_DIR" "$CTRL_DIR"

  STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "================================================================"
  echo "SEED $seed  started $STARTED"
  echo "================================================================"

  echo "[seed $seed] stage 1: autonomous collision 27 -> 26"
  PYTHONUNBUFFERED=1 python3 collision_search_3x3.py \
    --mode demo \
    --seed "$seed" \
    --out "$COLL_DIR" \
    2>&1 | tee "$SEED_DIR/collision.log"
  collision_rc=${PIPESTATUS[0]}

  controller_rc=99
  if (( collision_rc == 0 )) && [[ -f "$COLL_DIR/rank26.pt" ]]; then
    echo "[seed $seed] stage 2: specialist Pareto beam 26 -> 23"
    PYTHONUNBUFFERED=1 python3 autonomous_state_machine_3x3.py \
      --start "$COLL_DIR/rank26.pt" \
      --goal-rank 23 \
      --seed "$seed" \
      --max-cycles "$MAX_CYCLES" \
      --out "$CTRL_DIR" \
      2>&1 | tee "$SEED_DIR/controller.log"
    controller_rc=${PIPESTATUS[0]}
  else
    echo "[seed $seed] collision stage failed or did not create rank26.pt; controller skipped" | tee "$SEED_DIR/controller.log"
  fi

  FINISHED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$seed" "$collision_rc" "$controller_rc" "$STARTED" "$FINISHED" \
    >> "$OUT_ROOT/run_status.tsv"

  # Refresh human-readable summary after every seed, so partial overnight runs
  # are already useful if the job is interrupted.
  python3 "$SUMMARY_SCRIPT" "$OUT_ROOT" \
    --out-md "$OUT_ROOT/SUMMARY.md" \
    --out-csv "$OUT_ROOT/summary.csv" || true

done

python3 "$SUMMARY_SCRIPT" "$OUT_ROOT" \
  --out-md "$OUT_ROOT/SUMMARY.md" \
  --out-csv "$OUT_ROOT/summary.csv"

echo
echo "Finished. Summary: $OUT_ROOT/SUMMARY.md"
