#!/usr/bin/env python3
"""Which ground-truth objects does a run find? Recall per AV2 category and per range bin.

Per frame, Hungarian matching of output boxes to visible GT (num_interior_pts > 0)
within 50 m, accepted at <= 1 m BEV centre distance (CLEAR threshold). Ego frame,
no ROI filter, so the totals differ from score.py's; this is a breakdown, not the metric.

  .venv-eval/bin/python -P scripts/category_recall.py --run runs/final_test
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as pf
from scipy.optimize import linear_sum_assignment

RANGE_BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--av2", type=Path, default=Path("data/av2/val"))
    ap.add_argument("--min-count", type=int, default=200, help="categories with fewer GT boxes go to OTHER")
    args = ap.parse_args()

    n_gt, n_hit = Counter(), Counter()
    r_gt, r_hit = Counter(), Counter()
    for log in sorted(p for p in args.run.iterdir() if (p / "detections.csv").exists()):
        a = pf.read_table(args.av2 / log.name / "annotations.feather").to_pydict()
        a = {k: np.asarray(v) for k, v in a.items()}
        rng = np.hypot(a["tx_m"], a["ty_m"])
        keep = (a["num_interior_pts"] > 0) & (rng <= 50)
        a = {k: v[keep] for k, v in a.items()}
        rng = rng[keep]
        det = defaultdict(list)
        with open(log / "detections.csv") as f:
            for r in csv.DictReader(f):
                det[int(r["stamp_ns"])].append((float(r["cx"]), float(r["cy"])))
        for ts in np.unique(a["timestamp_ns"]):
            m = np.where(a["timestamp_ns"] == ts)[0]
            hit = np.zeros(len(m), bool)
            d_xy = np.array(det.get(int(ts), []))
            if len(d_xy):
                g = np.stack([a["tx_m"][m], a["ty_m"][m]], 1)
                d = np.linalg.norm(g[:, None] - d_xy[None], axis=2)
                rows, cols = linear_sum_assignment(np.where(d <= 1.0, d, 1e6))
                hit[rows[d[rows, cols] <= 1.0]] = True
            for k, i in enumerate(m):
                c = a["category"][i]
                n_gt[c] += 1
                n_hit[c] += hit[k]
                b = next(j for j, (lo, hi) in enumerate(RANGE_BINS) if lo <= rng[i] <= hi)
                r_gt[b] += 1
                r_hit[b] += hit[k]

    cats = {}
    other_gt = other_hit = 0
    for c, n in n_gt.most_common():
        if n >= args.min_count:
            cats[c] = {"gt": n, "recall": n_hit[c] / n}
        else:
            other_gt += n
            other_hit += n_hit[c]
    if other_gt:
        cats["OTHER"] = {"gt": other_gt, "recall": other_hit / other_gt}
    ranges = {f"{lo}-{hi} m": {"gt": r_gt[j], "recall": r_hit[j] / max(r_gt[j], 1)} for j, (lo, hi) in enumerate(RANGE_BINS)}
    total = {"gt": sum(n_gt.values()), "recall": sum(n_hit.values()) / max(sum(n_gt.values()), 1)}
    out = {"run": str(args.run), "total": total, "by_category": cats, "by_range": ranges}
    (args.run / "category_recall.json").write_text(json.dumps(out, indent=2))
    print(f"{'category':26s} {'GT boxes':>9s} {'recall':>7s}")
    for c, v in cats.items():
        print(f"{c:26s} {v['gt']:9d} {v['recall']:7.1%}")
    print(f"{'TOTAL':26s} {total['gt']:9d} {total['recall']:7.1%}\n")
    for k, v in ranges.items():
        print(f"{k:26s} {v['gt']:9d} {v['recall']:7.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
