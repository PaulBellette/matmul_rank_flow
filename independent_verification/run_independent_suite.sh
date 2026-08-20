#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-results/replication_5seeds}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Numerical polished endpoints
for s in 101 211 307 401 503; do
  p="$ROOT/endpoint_analysis/polished/seed_${s}_rank23_refined.pt"
  if [[ -f "$p" ]]; then
    echo "===== numerical seed $s ====="
    python3 "$HERE/verify_numerical_operational.py" "$p"
  fi
done

# Exact certificates
for s in 101 211 401 307 503; do
  p="$ROOT/exact_endpoints/seed_${s}/exact/rank23_exact.json"
  if [[ -f "$p" ]]; then
    echo "===== SymPy exact seed $s ====="
    python3 "$HERE/verify_sympy_exact.py" "$p"
    # Rational verifier applies only when the certificate says degree 1.
    degree=$(python3 - "$p" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x.get('field_degree',x.get('number_field',{}).get('degree','')))
PY
)
    if [[ "$degree" == "1" ]]; then
      echo "===== Fraction exact seed $s ====="
      python3 "$HERE/verify_rational_standalone.py" "$p"
    fi
  fi
done
