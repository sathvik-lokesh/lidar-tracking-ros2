#!/usr/bin/env python3
"""Check that every number in README.md is re-derived from results/results.json.

Each check formats a value from results.json (or a value derived from it) exactly as
the README prints it and asserts that the README contains that text. Table rows are
checked as whole cell sequences, so a number in the wrong row fails too.

  python3 -P scripts/verify_readme_numbers.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

R = json.loads(Path("results/results.json").read_text())
README = Path("README.md").read_text()
S, T, M = R["scores"], R["timing"], R["monitor"]
checks: list[tuple[str, str]] = []


def f3(x): return f"{x:.3f}"
def pct(x, d=1): return f"{100 * x:.{d}f}%"
def n(x): return f"{x:,}"


def row(label, *cells):
    checks.append((label, " | ".join(cells)))


def has(label, text):
    checks.append((label, text))


# Headline
has("headline HOTA", f"HOTA **{f3(S['test_final']['HOTA'])}**")
has("headline GT", f"{n(S['test_final']['gt_boxes_evaluated'])} evaluated ground-truth boxes")
has("headline ms", f"at {T['test_final']['mean_ms']:.1f} ms per frame mean")

# Scorer checks
for key, label in [("all_oracle_gt", "Ground truth with its real track ids"),
                   ("all_oracle_gt_noid", "Ground truth with a new id in every frame")]:
    s = S[key]
    row(label, label, f3(s["HOTA"]), f3(s["DetA"]), f3(s["AssA"]))
has("noid ceiling text", "perfect detection alone stops near 0.12")
has("upstream roi kept", f"keeps {n(S['all_notrack_upstream_roi']['gt_boxes_evaluated'])} ground-truth boxes")
has("city roi kept", f"city frame keeps {n(S['all_notrack_city_roi']['gt_boxes_evaluated'])}")

# Test ladder
ladder = [("test_notrack_session3", "First detector (PCL KD-tree clustering, length and area limits only), no tracking"),
          ("test_notrack_final_det", "Final detector, no tracking"),
          ("test_nn", "+ nearest-neighbour association"),
          ("test_kf_hits1", "+ Kalman filter and Hungarian assignment"),
          ("test_final", "**+ report after 3 hits (final)**"),
          ("test_oracle_ground", "Final, with the map's ground height instead of ours (oracle)"),
          ("test_ceiling_notrack", "Ceiling without tracking (ground truth, new id each frame)")]
for key, label in ladder:
    s = S[key]
    hota = f"**{f3(s['HOTA'])}**" if key == "test_final" else f3(s["HOTA"])
    row(label, label, hota, f3(s["DetA"]), f3(s["AssA"]), f3(s["DetPr"]), f3(s["DetRe"]), n(s["IDSW"]))
has("dev final", f"scores {f3(S['dev_final']['HOTA'])} on dev")
has("drop to test", f"drop to {f3(S['test_final']['HOTA'])} on test")
has("ceiling text", f"would stop at {f3(S['test_ceiling_notrack']['HOTA'])}")
has("nn jump", f"from {f3(S['test_notrack_final_det']['HOTA'])} to {f3(S['test_nn']['HOTA'])}")
has("kf+confirm gain", f"add another {f3(S['test_final']['HOTA'] - S['test_nn']['HOTA'])}")
has("oracle gain", f"gains {f3(S['test_oracle_ground']['HOTA'] - S['test_final']['HOTA'])}")
has("detpr text", f"Detection precision is {f3(S['test_final']['DetPr'])}")

# Category recall
cr = R["category_recall_test"]
for c in ["REGULAR_VEHICLE", "PEDESTRIAN", "CONSTRUCTION_BARREL", "BOLLARD", "STOP_SIGN"]:
    v = cr["by_category"][c]
    row(c, c, n(v["gt"]), pct(v["recall"]))
big = [cr["by_category"][c] for c in ("BUS", "BOX_TRUCK", "LARGE_VEHICLE")]
row("big vehicles", "BUS / BOX_TRUCK / LARGE_VEHICLE", " / ".join(n(v["gt"]) for v in big),
    " / ".join(pct(v["recall"]) for v in big))
row("all visible", "All visible boxes within 50 m", n(cr["total"]["gt"]), pct(cr["total"]["recall"]))
row("range recall", "Recall", *[pct(v["recall"]) for v in cr["by_range"].values()])

# Clustering method (dev)
for key, tkey, label in [("dev_ladder_kf_h3_pcl", "dev_kf_h3_pcl", "PCL voxel grid + KD-tree Euclidean clustering"),
                         ("dev_ladder_kf_h3_grid", "dev_kf_h3_grid", "BEV grid connected components")]:
    t = T[tkey]
    row(label, label, f3(S[key]["HOTA"]),
        f"{t['mean_ms']:.1f} / {t['p95_ms']:.1f} / {t['p99_ms']:.1f} ms", pct(t["over_100ms"]))
has("speedup", f"is {T['dev_kf_h3_pcl']['mean_ms'] / T['dev_kf_h3_grid']['mean_ms']:.0f}x faster")
has("merge bound", f"up to {2 * 0.15 * math.sqrt(2):.2f} m further apart")

# Runtime
t = T["test_final"]
for key, label in [("ground_ms", "Ground removal"), ("cluster_ms", "Clustering"), ("fit_ms", "Box fitting"), ("track_ms", "Tracker")]:
    row(label, label, f"{t[key]:.2f} ms")
has("runtime total", f"**{t['mean_ms']:.1f} ms mean, {t['p99_ms']:.1f} ms p99, {t['max_ms']:.1f} ms max** ({n(t['frames'])} frames")

# Monitor (test)
mt = M["test"]
n_err = sum(mt["errors"].values())
has("monitor header", f"Test split ({n(mt['reports_with_predecessor'])} reports, {n_err} errors)")
for k, label in [("k0", "Same report"), ("k5", "Within the 5 reports (0.5 s) before")]:
    x = mt[k]
    row(label, label, pct(x["recall_all"]), pct(x["recall_chance"]), pct(x["p_err_given_flag"], 2), pct(x["p_err_given_no_flag"], 2))
has("monitor ratio", f"at about {mt['k0']['recall_all'] / mt['k0']['recall_chance']:.1f}x the chance rate")
has("monitor flags without error", f"{100 * (1 - mt['k0']['p_err_given_flag']):.0f}% of its flags")
speed_rate = mt["flag_rate_by_type"]["speed"]
if speed_rate != 0.0:
    checks.append(("speed never fired", "__speed flag rate is not zero__"))

# Live pipeline
lp = R["live_pipeline"]
has("live frames", f"`/tracks` for {lp['frames_received']} of the {lp['frames_in_bag']} frames")
has("live ids", f"same track ids as `process_bag` in {lp['frames_identical_track_ids']} of {lp['frames_in_bag']} frames")
has("live flags", f"same {lp['monitor_flags_live']} monitor flags ({lp['monitor_flags_offline']} offline)")
for node, label in [("lidar_detector", "lidar_detector"), ("lidar_tracker", "lidar_tracker"), ("track_monitor", "track_monitor")]:
    d = lp["diagnostics"][f"{node}: processing time"]
    row(label, label, f"{float(d['mean_ms']):.2f} ms", f"{float(d['p95_ms']):.2f} ms", f"{float(d['budget_ms']):.0f} ms")

failed = [(label, text) for label, text in checks if text not in README]
for label, text in failed:
    print(f"FAIL {label}: README lacks '{text}'")
print(f"{len(checks) - len(failed)}/{len(checks)} README numbers verified")
sys.exit(1 if failed else 0)
