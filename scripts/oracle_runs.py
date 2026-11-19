#!/usr/bin/env python3
"""Write ground truth out as fake runs, to check that score.py scores what it should.

  runs/oracle_gt/<log>/detections.csv        GT boxes with their real track ids  -> expect HOTA = 1
  runs/oracle_gt_noid/<log>/detections.csv   GT boxes, fresh id every frame      -> expect DetA = 1, AssA << 1
  runs/oracle_gt_shift1m/<log>/detections.csv GT boxes moved 1 m in x (ego frame) -> expect LocA ~ 0.5

Only cuboids with lidar points are written, matching what score.py evaluates.
Boxes are written in the ego frame with the same columns as process_bag.

  .venv-eval/bin/python -P scripts/oracle_runs.py --av2 data/av2/val --logs 02678d04-...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.feather as pf

HEADER = "stamp_ns,track_id,cx,cy,cz,length,width,height,yaw,num_points\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--av2", type=Path, default=Path("data/av2/val"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--logs", nargs="+", required=True)
    args = ap.parse_args()

    for log_id in args.logs:
        a = pf.read_table(args.av2 / log_id / "annotations.feather").to_pydict()
        a = {k: np.asarray(v) for k, v in a.items()}
        a = {k: v[a["num_interior_pts"] > 0] for k, v in a.items()}
        yaw = np.arctan2(2 * (a["qw"] * a["qz"] + a["qx"] * a["qy"]), 1 - 2 * (a["qy"] ** 2 + a["qz"] ** 2))
        uuid_to_int = {u: i for i, u in enumerate(sorted(set(a["track_uuid"].tolist())))}
        for variant in ("oracle_gt", "oracle_gt_noid", "oracle_gt_shift1m"):
            out = args.runs / variant / log_id
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "detections.csv", "w") as f:
                f.write(HEADER)
                for i in range(len(yaw)):
                    tid = uuid_to_int[a["track_uuid"][i]] if variant != "oracle_gt_noid" else i
                    dx = 1.0 if variant == "oracle_gt_shift1m" else 0.0
                    f.write(
                        f"{a['timestamp_ns'][i]},{tid},{a['tx_m'][i] + dx:.4f},{a['ty_m'][i]:.4f},{a['tz_m'][i]:.4f},"
                        f"{a['length_m'][i]:.4f},{a['width_m'][i]:.4f},{a['height_m'][i]:.4f},{yaw[i]:.5f},"
                        f"{a['num_interior_pts'][i]}\n")
        print(log_id, "->", len(yaw), "boxes x 3 variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
