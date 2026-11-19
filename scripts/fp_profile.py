#!/usr/bin/env python3
"""Profile true vs false positives of a run, to choose false-positive gates on DEV only.

A box counts as TP if a visible GT cuboid (num_interior_pts > 0) lies within 1 m
(BEV centre, ego frame) in the same frame, greedily one-to-one, else FP. Only boxes
within 50 m are counted. This is a diagnostic, not the reported metric (score.py is).

  .venv/bin/python -P scripts/fp_profile.py --run runs/dev_none
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.feather as pf

FEATURES = ["num_points", "height", "length", "width", "area", "cz", "clearance", "range", "hits"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--av2", type=Path, default=Path("data/av2/val"))
    args = ap.parse_args()

    rows = {f: {"TP": [], "FP": []} for f in FEATURES}
    for log in sorted(p for p in args.run.iterdir() if (p / "detections.csv").exists()):
        a = pf.read_table(args.av2 / log.name / "annotations.feather").to_pydict()
        a = {k: np.asarray(v) for k, v in a.items()}
        a = {k: v[a["num_interior_pts"] > 0] for k, v in a.items()}
        gt: dict[int, np.ndarray] = {}
        for ts in np.unique(a["timestamp_ns"]):
            m = a["timestamp_ns"] == ts
            gt[int(ts)] = np.stack([a["tx_m"][m], a["ty_m"][m]], axis=1)
        dets: dict[int, list[dict]] = {}
        with open(log / "detections.csv") as f:
            for r in csv.DictReader(f):
                dets.setdefault(int(r["stamp_ns"]), []).append(r)
        for ts, rs in dets.items():
            g = gt.get(ts, np.zeros((0, 2)))
            used = np.zeros(len(g), bool)
            xy = np.array([[float(r["cx"]), float(r["cy"])] for r in rs])
            d = np.linalg.norm(xy[:, None, :] - g[None, :, :], axis=2) if len(g) else np.full((len(rs), 0), 9e9)
            order = np.argsort(d.min(axis=1) if len(g) else np.zeros(len(rs)))
            for i in order:
                r = rs[i]
                rng = float(np.hypot(*xy[i]))
                if rng > 50:
                    continue
                label = "FP"
                if len(g):
                    cand = np.where(~used & (d[i] < 1.0))[0]
                    if len(cand):
                        used[cand[np.argmin(d[i][cand])]] = True
                        label = "TP"
                L, W = float(r["length"]), float(r["width"])
                vals = {"num_points": float(r["num_points"]), "height": float(r["height"]), "length": L,
                        "width": W, "area": L * W, "cz": float(r["cz"]), "range": rng,
                        "hits": float(r.get("hits", 1) or 1), "clearance": float(r.get("clearance", 0) or 0)}
                for k in FEATURES:
                    rows[k][label].append(vals[k])

    n_tp, n_fp = len(rows["height"]["TP"]), len(rows["height"]["FP"])
    print(f"{args.run}: TP {n_tp}  FP {n_fp}  (within 50 m, 1 m match, no ROI filter)")
    print(f"{'feature':11s} {'TP p10':>8s} {'TP p50':>8s} {'TP p90':>8s} | {'FP p10':>8s} {'FP p50':>8s} {'FP p90':>8s}")
    for k in FEATURES:
        tp, fp = np.array(rows[k]["TP"]), np.array(rows[k]["FP"])
        q = lambda v, p: np.percentile(v, p) if len(v) else float("nan")  # noqa: E731
        print(f"{k:11s} {q(tp,10):8.2f} {q(tp,50):8.2f} {q(tp,90):8.2f} | {q(fp,10):8.2f} {q(fp,50):8.2f} {q(fp,90):8.2f}")
    # What a single threshold would do: share of TP kept vs FP removed.
    print("\nsingle-threshold gates (keep if feature >= t):")
    for k, ts in [("num_points", [3, 5, 8, 12]), ("height", [0.3, 0.5, 0.8, 1.0]), ("width", [0.1, 0.2, 0.3]), ("hits", [2, 3, 5])]:
        tp, fp = np.array(rows[k]["TP"]), np.array(rows[k]["FP"])
        for t in ts:
            print(f"  {k:10s} >= {t:<4}  keeps TP {np.mean(tp >= t):5.1%}  removes FP {np.mean(fp < t):5.1%}")
    print("\nclearance gates (keep if clearance <= t):")
    tp, fp = np.array(rows["clearance"]["TP"]), np.array(rows["clearance"]["FP"])
    for t in [0.3, 0.5, 0.8, 1.2]:
        print(f"  clearance  <= {t:<4}  keeps TP {np.mean(tp <= t):5.1%}  removes FP {np.mean(fp > t):5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
