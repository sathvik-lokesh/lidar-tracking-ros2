#!/usr/bin/env python3
"""Per-frame runtime of a run (detector + tracker), from <run>/<log>/timings.csv.

Single-threaded process_bag timings; message (de)serialisation and bag I/O excluded.

  .venv/bin/python -P scripts/timing_summary.py runs/dev_kf_h3_grid [more runs...]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

STAGES = ["ground_ms", "voxel_ms", "cluster_ms", "fit_ms", "track_ms"]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for run in map(Path, sys.argv[1:]):
        rows = []
        for f in sorted(run.glob("*/timings.csv")):
            rows += list(csv.DictReader(open(f)))
        if not rows:
            print(run, "no timings")
            continue
        total = np.array([float(r["detect_ms"]) + float(r.get("track_ms", 0) or 0) for r in rows])
        parts = "  ".join(
            f"{s[:-3]} {np.mean([float(r.get(s, 0) or 0) for r in rows]):.1f}" for s in STAGES)
        print(f"{run.name}: frames {len(total)}  mean {total.mean():.1f}  p50 {np.percentile(total, 50):.1f}  "
              f"p95 {np.percentile(total, 95):.1f}  p99 {np.percentile(total, 99):.1f}  max {total.max():.1f} ms"
              f"  | over 100 ms: {np.mean(total > 100):.1%}  | mean per stage: {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
