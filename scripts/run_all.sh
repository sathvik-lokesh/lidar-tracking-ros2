#!/usr/bin/env bash
# Run process_bag over the logs of one split, then score the run.
#   scripts/run_all.sh <dev|test|all> <run_name> [key=value ...]
#   e.g. scripts/run_all.sh dev kf_v1 tracker=kf min_hits=3
set -euo pipefail
cd "$(dirname "$0")/.."
split=${1:?usage: run_all.sh <dev|test|all> <run_name> [key=value ...]}
run=${2:?usage: run_all.sh <dev|test|all> <run_name> [key=value ...]}
shift 2
set +u; source env.sh; set -u  # ROS setup scripts read unset variables
bin=install/lidar_tracking/lib/lidar_tracking/process_bag
logs=$(/usr/bin/python3 -c "import json,sys; s=json.load(open('splits.json')); print(' '.join(s['dev']+s['test'] if sys.argv[1]=='all' else s[sys.argv[1]]))" "$split")
mkdir -p "runs/$run"
for log in $logs; do
  printf '%s ' "${log:0:8}"
  "$bin" "data/bags/$log" "runs/$run/$log" "$@"
done
.venv-eval/bin/python -P scripts/score.py --run "runs/$run" 2>&1 | grep -v -E "zero_distance|BURST"
