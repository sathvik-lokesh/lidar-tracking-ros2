#!/usr/bin/env bash
# Run process_bag over every bag, then score the run.
#   scripts/run_all.sh <run_name> [key=value ...]     e.g. scripts/run_all.sh det_v1 cluster_tolerance_m=0.7
set -euo pipefail
cd "$(dirname "$0")/.."
run=${1:?usage: run_all.sh <run_name> [key=value ...]}
shift
set +u; source env.sh; set -u  # ROS setup scripts read unset variables
bin=install/lidar_tracking/lib/lidar_tracking/process_bag
mkdir -p "runs/$run"
for bag in data/bags/*/; do
  log=$(basename "$bag")
  printf '%s ' "$log"
  "$bin" "$bag" "runs/$run/$log" "$@"
done
.venv-eval/bin/python -P scripts/score.py --run "runs/$run" 2>&1 | grep -v -E "zero_distance|BURST"
