#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
PYTHONUNBUFFERED=1 python3 run_rank22_campaign.py "$@" 2>&1 | tee results/rank22_campaign.log
