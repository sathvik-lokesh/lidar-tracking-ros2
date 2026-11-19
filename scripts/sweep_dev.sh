#!/usr/bin/env bash
# One-factor-at-a-time parameter sweep on the DEV split only. Each line of the
# config file is "<name> key=value ...", added on top of the base config.
#   scripts/sweep_dev.sh <configs.txt> "<base key=value ...>"
# Writes a table to runs/sweep_<configs-basename>.tsv (and stdout).
set -euo pipefail
cd "$(dirname "$0")/.."
cfgs=${1:?usage: sweep_dev.sh <configs.txt> "<base params>"}
base=${2:-}
out="runs/sweep_$(basename "${cfgs%.*}").tsv"
printf 'name\tHOTA\tDetA\tAssA\tDetRe\tDetPr\tIDSW\tparams\n' | tee "$out"
while read -r name params; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  # shellcheck disable=SC2086
  scripts/run_all.sh dev "sw_$name" $base $params > /dev/null
  /usr/bin/python3 - "runs/sw_$name/score.json" "$name" "$base $params" <<'EOF' | tee -a "$out"
import json, sys
s = json.load(open(sys.argv[1]))
print("\t".join([sys.argv[2]] + [f"{s[k]:.4f}" for k in ("HOTA", "DetA", "AssA", "DetRe", "DetPr")] + [str(s["IDSW"]), sys.argv[3].strip()]))
EOF
done < "$cfgs"
