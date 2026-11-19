#!/usr/bin/env python3
"""Does the runtime monitor fire before tracking errors?

Matching: per frame, Hungarian on BEV centre distance between output tracks and
visible GT (num_interior_pts > 0) within 50 m of the ego, accepted if <= 1 m (the
CLEAR threshold score.py uses: similarity 1 - d/2 >= 0.5). Ego frame, no ROI filter.

Error events, per track report (a track's output in one frame):
  id_switch  matched to GT g now, and the track's last matched GT was a different g'
  lost       matched at the track's previous report, unmatched now
Thresholds of the monitor are a-priori (monitor.hpp), nothing here is tuned.

Reported, for windows k = 0 and k = 5 reports (0.5 s at 10 Hz):
  recall    share of error events with a flag on that track at the same report or in
            the k reports before it ("fired before the error")
  P(err|flag), P(err|no flag), lift: is an error (on that track, at this report or
            in the next k reports) likelier after a flag than without one?
Only reports that have a previous report of the same track are counted (the
monitor cannot fire on a track's first report).
Chance baseline: the same flags shuffled across all counted reports (same flag
rate, no relation to errors), averaged over 20 shuffles. A monitor is only useful
where it beats this.

  .venv-eval/bin/python -P scripts/monitor_eval.py --run runs/final_test
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as pf
from scipy.optimize import linear_sum_assignment

MATCH_M = 1.0
RANGE_M = 50.0
FLAG_NAMES = {1: "speed", 2: "small_speed", 4: "size"}


def load_gt(log_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    a = pf.read_table(log_dir / "annotations.feather").to_pydict()
    a = {k: np.asarray(v) for k, v in a.items()}
    keep = (a["num_interior_pts"] > 0) & (np.hypot(a["tx_m"], a["ty_m"]) <= RANGE_M)
    a = {k: v[keep] for k, v in a.items()}
    out = {}
    for ts in np.unique(a["timestamp_ns"]):
        m = a["timestamp_ns"] == ts
        out[int(ts)] = (np.stack([a["tx_m"][m], a["ty_m"][m]], 1), a["track_uuid"][m])
    return out


def track_reports(run_log: Path, gt) -> dict[str, list[dict]]:
    """track id -> chronological list of {flag, gt} (gt = matched GT uuid or None)."""
    frames = defaultdict(list)
    with open(run_log / "detections.csv") as f:
        for r in csv.DictReader(f):
            frames[int(r["stamp_ns"])].append(r)
    tracks = defaultdict(list)
    for ts in sorted(frames):
        rs = [r for r in frames[ts] if np.hypot(float(r["cx"]), float(r["cy"])) <= RANGE_M]
        if not rs:
            continue
        xy = np.array([[float(r["cx"]), float(r["cy"])] for r in rs])
        matched = [None] * len(rs)
        if ts in gt:
            gxy, gid = gt[ts]
            d = np.linalg.norm(xy[:, None] - gxy[None], axis=2)
            rows, cols = linear_sum_assignment(np.where(d <= MATCH_M, d, 1e6))
            for i, j in zip(rows, cols):
                if d[i, j] <= MATCH_M:
                    matched[i] = gid[j]
        for r, g in zip(rs, matched):
            tracks[r["track_id"]].append({"flag": int(r["flags"]), "gt": g})
    return tracks


def label_errors(reports: list[dict]) -> list[str | None]:
    errs: list[str | None] = [None] * len(reports)
    last_gt = None
    for i, r in enumerate(reports):
        if i > 0:
            if r["gt"] is not None and last_gt is not None and r["gt"] != last_gt:
                errs[i] = "id_switch"
            elif r["gt"] is None and reports[i - 1]["gt"] is not None:
                errs[i] = "lost"
        if r["gt"] is not None:
            last_gt = r["gt"]
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--av2", type=Path, default=Path("data/av2/val"))
    args = ap.parse_args()

    all_tracks = []
    for log in sorted(p for p in args.run.iterdir() if (p / "detections.csv").exists()):
        for reps in track_reports(log, load_gt(args.av2 / log.name)).values():
            all_tracks.append((reps, label_errors(reps)))

    summary = {"run": str(args.run)}
    n_reports = sum(len(r) - 1 for r, _ in all_tracks)
    flags = [r[i]["flag"] for r, _ in all_tracks for i in range(1, len(r))]
    summary["reports_with_predecessor"] = n_reports
    summary["flag_rate"] = float(np.mean([f != 0 for f in flags]))
    summary["flag_rate_by_type"] = {n: float(np.mean([(f & b) != 0 for f in flags])) for b, n in FLAG_NAMES.items()}
    n_err = defaultdict(int)
    for _, e in all_tracks:
        for x in e:
            if x:
                n_err[x] += 1
    summary["errors"] = dict(n_err)

    rng = np.random.default_rng(0)

    def recall_for(flag_lists, k):
        caught = 0
        for (reps, errs), fl in zip(all_tracks, flag_lists):
            for i in range(1, len(reps)):
                if errs[i] and any(fl[max(1, i - k): i + 1]):
                    caught += 1
        return caught / max(sum(n_err.values()), 1)

    def shuffled_flags():
        pool = rng.permutation(np.array(flags) != 0)
        out, pos = [], 0
        for reps, _ in all_tracks:
            n = len(reps) - 1
            out.append([False] + list(pool[pos: pos + n]))
            pos += n
        return out

    for k in (0, 5):
        caught = defaultdict(int)
        e_after_f = e_after_nf = n_f = n_nf = 0
        for reps, errs in all_tracks:
            fl = [r["flag"] != 0 for r in reps]
            for i in range(1, len(reps)):
                if errs[i] and any(fl[max(1, i - k): i + 1]):
                    caught[errs[i]] += 1
                err_soon = any(errs[i: i + k + 1])
                if fl[i]:
                    n_f += 1
                    e_after_f += err_soon
                else:
                    n_nf += 1
                    e_after_nf += err_soon
        p_f = e_after_f / max(n_f, 1)
        p_nf = e_after_nf / max(n_nf, 1)
        summary[f"k{k}"] = {
            "recall_by_error": {e: caught[e] / max(n_err[e], 1) for e in n_err},
            "recall_all": sum(caught.values()) / max(sum(n_err.values()), 1),
            "recall_chance": float(np.mean([recall_for(shuffled_flags(), k) for _ in range(20)])),
            "p_err_given_flag": p_f,
            "p_err_given_no_flag": p_nf,
            "lift": p_f / p_nf if p_nf else float("nan"),
        }
    out = args.run / "monitor_eval.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
