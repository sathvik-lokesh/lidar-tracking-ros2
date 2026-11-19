#!/usr/bin/env python3
"""Ground-removal oracle from the AV2 map: per-sweep keep masks.

For each sweep, every point is moved to the city frame with the ego pose, the map's
ground height (av2 GroundHeightLayer, 30 cm raster) is looked up, and the point is
kept if it lies in the same band the detector uses: ground_clearance_m < h <=
max_height_m above ground. Points with no raster value (NaN) are dropped.

Output: <out>/<log>/<timestamp_ns>.bin, one byte per point (1 = keep), in the
feather (= bag) point order. process_bag reads them with ground_mask_dir=<out>.

  .venv-eval/bin/python -P scripts/oracle_ground_masks.py --logs <log ids...>
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pyarrow.feather as pf
from scipy.spatial.transform import Rotation

warnings.filterwarnings("ignore", category=FutureWarning)
from av2.map.map_api import GroundHeightLayer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--av2", type=Path, default=Path("data/av2/val"))
    ap.add_argument("--out", type=Path, default=Path("data/oracle_ground_masks"))
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--clearance", type=float, default=0.25)
    ap.add_argument("--max-height", type=float, default=3.0)
    args = ap.parse_args()

    for log_id in args.logs:
        log = args.av2 / log_id
        ground = GroundHeightLayer.from_file(log / "map")
        poses = pf.read_table(log / "city_SE3_egovehicle.feather").to_pydict()
        pose = {ts: i for i, ts in enumerate(poses["timestamp_ns"])}
        out = args.out / log_id
        out.mkdir(parents=True, exist_ok=True)
        kept = total = nan = 0
        for f in sorted((log / "sensors" / "lidar").glob("*.feather")):
            ts = int(f.stem)
            i = pose[ts]
            rot = Rotation.from_quat([poses["qx"][i], poses["qy"][i], poses["qz"][i], poses["qw"][i]])
            t = np.array([poses["tx_m"][i], poses["ty_m"][i], poses["tz_m"][i]])
            tb = pf.read_table(f, columns=["x", "y", "z"])
            xyz = np.stack([tb[c].to_numpy().astype(np.float32).astype(np.float64) for c in "xyz"], 1)
            city = rot.apply(xyz) + t
            g = ground.get_ground_height_at_xy(city)
            h = city[:, 2] - g
            keep = np.isfinite(h) & (h > args.clearance) & (h <= args.max_height)
            keep.astype(np.uint8).tofile(out / f"{ts}.bin")
            kept += int(keep.sum())
            total += len(keep)
            nan += int((~np.isfinite(g)).sum())
        print(f"{log_id[:8]}: kept {kept / total:.1%} of points, no-raster {nan / total:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
