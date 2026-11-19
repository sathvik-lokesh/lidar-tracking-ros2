#!/usr/bin/env python3
"""Collect every number the README reports into results/results.json.

Sources are the run folders under runs/ (score.json, timings.csv, monitor_eval.json,
category_recall.json) and the live-pipeline record. Run after regenerating the runs;
then scripts/verify_readme_numbers.py checks the README against this file.

  python3 -P scripts/make_results.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

RUNS = Path("runs")
METRICS = ["HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "MOTA", "IDSW", "gt_boxes_evaluated"]

# name in results.json -> run folder
SCORED = {
    # test ladder (run once, config frozen beforehand)
    "test_notrack_session3": "test_notrack_pcl",
    "test_notrack_final_det": "test_notrack_det",
    "test_nn": "test_nn",
    "test_kf_hits1": "test_kf_h1",
    "test_final": "final_test",
    "test_oracle_ground": "oracleground_test",
    "test_ceiling_notrack": "oracle_gt_noid_test",
    # dev
    "dev_final": "final_dev",
    "dev_oracle_ground": "oracleground_dev",
    "dev_ladder_none_pcl": "dev_none",
    "dev_ladder_nn_pcl": "dev_nn",
    "dev_ladder_kf_pcl": "dev_kf",
    "dev_ladder_kf_h3_pcl": "dev_kf_h3_pcl",
    "dev_ladder_kf_h3_grid": "dev_kf_h3_grid",
    # scorer checks on all 20 logs
    "all_oracle_gt": "oracle_gt",
    "all_oracle_gt_noid": "oracle_gt_noid",
    "all_notrack_upstream_roi": None,  # filled from score_upstream_roi.json below
    "all_notrack_city_roi": "notrack_v1",
}


def timing(run: Path) -> dict:
    rows = []
    for f in sorted(run.glob("*/timings.csv")):
        rows += list(csv.DictReader(open(f)))
    tot = np.array([float(r["detect_ms"] if "detect_ms" in r else r["total_ms"]) + float(r.get("track_ms", 0) or 0)
                    for r in rows])
    stage = {s: float(np.mean([float(r[s]) for r in rows])) for s in ("ground_ms", "cluster_ms", "fit_ms") if s in rows[0]}
    if "track_ms" in rows[0]:
        stage["track_ms"] = float(np.mean([float(r["track_ms"]) for r in rows]))
    return {"frames": len(tot), "mean_ms": float(tot.mean()), "p50_ms": float(np.percentile(tot, 50)),
            "p95_ms": float(np.percentile(tot, 95)), "p99_ms": float(np.percentile(tot, 99)),
            "max_ms": float(tot.max()), "over_100ms": float(np.mean(tot > 100)), **stage}


def main() -> int:
    out: dict = {"scores": {}, "timing": {}}
    for name, folder in SCORED.items():
        if folder is None:
            continue
        s = json.loads((RUNS / folder / "score.json").read_text())
        out["scores"][name] = {k: s[k] for k in METRICS if k in s}
    s = json.loads((RUNS / "notrack_v1" / "score_upstream_roi.json").read_text())
    out["scores"]["all_notrack_upstream_roi"] = {k: s[k] for k in METRICS if k in s}

    for name, folder in [("test_final", "final_test"), ("test_notrack_session3", "test_notrack_pcl"),
                         ("dev_kf_h3_pcl", "dev_kf_h3_pcl"), ("dev_kf_h3_grid", "dev_kf_h3_grid"),
                         ("test_oracle_ground", "oracleground_test")]:
        out["timing"][name] = timing(RUNS / folder)

    out["monitor"] = {s: json.loads((RUNS / f"final_{s}" / "monitor_eval.json").read_text()) for s in ("dev", "test")}
    out["category_recall_test"] = json.loads((RUNS / "final_test" / "category_recall.json").read_text())
    # Live composed pipeline vs process_bag on the same log (same frozen config).
    log = "1f434d15-8745-3fba-9c3e-ccb026688397"
    live = json.loads((RUNS / "live_pipeline_1f434d15.json").read_text())
    off_ids, off_flags = {}, {}
    for r in csv.DictReader(open(RUNS / "live_ref" / log / "detections.csv")):
        off_ids.setdefault(r["stamp_ns"], []).append(r["track_id"])
        off_flags[r["stamp_ns"]] = off_flags.get(r["stamp_ns"], 0) + (int(r["flags"]) != 0)
    stamps = [r["stamp_ns"] for r in csv.DictReader(open(RUNS / "live_ref" / log / "timings.csv"))]
    lt, lf = live["tracks"], live["flags"]
    out["live_pipeline"] = {
        "frames_in_bag": len(stamps),
        "frames_received": sum(s in lt for s in stamps),
        "frames_identical_track_ids": sum(sorted(lt.get(s, []), key=int) == sorted(off_ids.get(s, []), key=int) for s in stamps),
        "monitor_flags_live": sum(lf.values()),
        "monitor_flags_offline": sum(off_flags.values()),
        "diagnostics": live["diagnostics"],
    }

    Path("results").mkdir(exist_ok=True)
    Path("results/results.json").write_text(json.dumps(out, indent=1))
    print("wrote results/results.json:", len(out["scores"]), "scored runs,", len(out["timing"]), "timing summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
