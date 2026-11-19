#!/usr/bin/env python3
"""Class-agnostic HOTA / CLEAR scoring of a run against Argoverse 2 ground truth.

Uses the official av2 0.3.6 tracking evaluation (TrackEval HOTA + CLEAR, BEV
centre-distance similarity with ZERO_DISTANCE = 2 m, 50 m range filter, drivable-area
ROI filter), with two deliberate differences, both stated in the README:

1. Class-agnostic: every GT category is renamed to OBJECT, and so is every prediction.
   A clustering pipeline has no class, so this is the fair question to ask of it.
2. No score-threshold tuning: all predictions carry score 1. The official `evaluate()`
   tunes a per-class threshold on the same data it reports on; we have nothing to tune.
3. Drivable-area ROI is tested in the city frame. av2 0.3.6 `filter_drivable_area` passes
   `translation_m - ego_translation_m` (a city-oriented offset) to
   `compute_objects_in_roi_mask`, which treats it as an ego-frame box and applies the full
   city_SE3_ego, so every box is rotated by the ego heading about the ego before the ROI
   lookup. Checked 2026-11-19: log 02678d04 (ego yaw ~115 deg) keeps 290 of 1531 GT boxes
   upstream vs 1515 when tested correctly; log 1f434d15 (|yaw| <= 33 deg) 7337 vs 8165.
   `--upstream-roi` reproduces the upstream behaviour for comparison.

GT cuboids with zero lidar points inside are dropped, mirroring the official AV2
detection eval (`compute_evaluated_gts_mask`: num_interior_pts > 0). A lidar-only
method cannot see them.

Run directory layout (written by process_bag): <run>/<log_id>/detections.csv (ego frame).

  .venv-eval/bin/python -P scripts/score.py --run runs/smoke --av2 data/av2/val
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pyarrow.feather as pf
from scipy.spatial.transform import Rotation

warnings.filterwarnings("ignore", category=FutureWarning)
from av2.evaluation.tracking import utils as av2_utils  # noqa: E402
from av2.evaluation.tracking.eval import (  # noqa: E402
    evaluate_tracking,
    filter_drivable_area,
    filter_max_dist,
)
from av2.map.map_api import ArgoverseStaticMap, RasterLayerType  # noqa: E402

CLASS = "OBJECT"
MAX_RANGE_M = 50


def load_poses(log_dir: Path) -> dict[int, tuple[Rotation, np.ndarray]]:
    t = pf.read_table(log_dir / "city_SE3_egovehicle.feather").to_pydict()
    out = {}
    for i, ts in enumerate(t["timestamp_ns"]):
        rot = Rotation.from_quat([t["qx"][i], t["qy"][i], t["qz"][i], t["qw"][i]])
        out[ts] = (rot, np.array([t["tx_m"][i], t["ty_m"][i], t["tz_m"][i]]))
    return out


def to_city(pose, xyz_ego: np.ndarray, yaw_ego: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot, trans = pose
    xyz = rot.apply(xyz_ego) + trans if len(xyz_ego) else np.zeros((0, 3))
    yaw = yaw_ego + rot.as_euler("xyz")[2]
    return xyz, yaw


def make_frame(log_id, ts, pose, ids, xyz_ego, size, yaw_ego) -> dict:
    n = len(ids)
    xyz, yaw = to_city(pose, xyz_ego, yaw_ego)
    return {
        "seq_id": log_id,
        "timestamp_ns": ts,
        "track_id": np.asarray(ids, dtype=object),
        "translation_m": xyz.reshape(n, 3),
        "size": np.asarray(size, dtype=float).reshape(n, 3),
        "yaw": np.asarray(yaw, dtype=float).reshape(n),
        "velocity_m_per_s": np.zeros((n, 3)),
        "label": np.zeros(n, dtype=int),
        "score": np.ones(n),
        "name": np.array([CLASS] * n, dtype=object),
        "ego_translation_m": pose[1].tolist(),  # a list on purpose: av2 masks every ndarray per object
    }


def quat_yaw(qw, qx, qy, qz) -> np.ndarray:
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def load_gt(log_dir: Path, poses) -> tuple[list[dict], int, int]:
    a = pf.read_table(log_dir / "annotations.feather").to_pydict()
    a = {k: np.asarray(v) for k, v in a.items()}
    n_all = len(a["timestamp_ns"])
    stamps = sorted(set(a["timestamp_ns"].tolist()))  # before filtering: keep frames whose boxes are all hidden
    keep = a["num_interior_pts"] > 0
    a = {k: v[keep] for k, v in a.items()}
    frames = []
    for ts in stamps:
        m = a["timestamp_ns"] == ts
        frames.append(make_frame(
            log_dir.name, ts, poses[ts], a["track_uuid"][m],
            np.stack([a["tx_m"][m], a["ty_m"][m], a["tz_m"][m]], axis=1),
            np.stack([a["length_m"][m], a["width_m"][m], a["height_m"][m]], axis=1),
            quat_yaw(a["qw"][m], a["qx"][m], a["qy"][m], a["qz"][m])))
    return frames, n_all, int(keep.sum())


def load_pred(run_log_dir: Path, log_id: str, poses, stamps: list[int]) -> list[dict]:
    rows: dict[int, list[dict]] = {ts: [] for ts in stamps}
    with open(run_log_dir / "detections.csv") as f:
        for r in csv.DictReader(f):
            ts = int(r["stamp_ns"])
            if ts in rows:
                rows[ts].append(r)
    frames = []
    for ts in stamps:
        rs = rows[ts]
        g = lambda k: np.array([float(r[k]) for r in rs])  # noqa: E731
        frames.append(make_frame(
            log_id, ts, poses[ts], [f"{log_id}:{r['track_id']}" for r in rs],
            np.stack([g("cx"), g("cy"), g("cz")], axis=1) if rs else np.zeros((0, 3)),
            np.stack([g("length"), g("width"), g("height")], axis=1) if rs else np.zeros((0, 3)),
            g("yaw")))
    return frames


def bev_corners(f: dict) -> np.ndarray:
    """(N, 4, 2) BEV corners of each box in the city frame."""
    t, s, y = f["translation_m"], f["size"], f["yaw"]
    c, sn = np.cos(y), np.sin(y)
    out = []
    for dx in (-0.5, 0.5):
        for dy in (-0.5, 0.5):
            u, v = dx * s[:, 0], dy * s[:, 1]
            out.append(np.stack([t[:, 0] + c * u - sn * v, t[:, 1] + sn * u + c * v], axis=1))
    return np.stack(out, axis=1)


def filter_roi_city(tracks: dict, av2_dir: Path) -> dict:
    """Keep boxes with any BEV corner inside the map ROI raster (same rule as av2)."""
    for log_id, frames in tracks.items():
        avm = ArgoverseStaticMap.from_map_dir(av2_dir / log_id / "map", build_raster=True)
        for i, f in enumerate(frames):
            if len(f["track_id"]) == 0:
                continue
            corners = bev_corners(f).reshape(-1, 2)
            keep = avm.get_raster_layer_points_boolean(corners, RasterLayerType.ROI).reshape(-1, 4).any(axis=1)
            frames[i] = av2_utils.index_array_values(f, keep)
    return tracks


def check_alignment(labels: dict, preds: dict) -> None:
    """Every (log, stamp) must exist in both, with the same ego pose."""
    lab = {(f["seq_id"], f["timestamp_ns"]): f["ego_translation_m"] for f in av2_utils.ungroup_frames(labels)}
    prd = {(f["seq_id"], f["timestamp_ns"]): f["ego_translation_m"] for f in av2_utils.ungroup_frames(preds)}
    assert lab.keys() == prd.keys(), "label and prediction frames differ"
    assert all(np.allclose(lab[k], prd[k]) for k in lab), "ego pose differs between labels and predictions"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--av2", type=Path, default=Path("data/av2/val"))
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/score.json")
    ap.add_argument("--upstream-roi", action="store_true", help="use av2 0.3.6 filter_drivable_area (yaw bug)")
    args = ap.parse_args()

    log_ids = sorted(p.name for p in args.run.iterdir() if (p / "detections.csv").exists())
    if not log_ids:
        print("no logs with detections.csv in", args.run, file=sys.stderr)
        return 1

    labels, preds = {}, {}
    gt_all = gt_visible = 0
    for log_id in log_ids:
        poses = load_poses(args.av2 / log_id)
        gt_frames, n_all, n_vis = load_gt(args.av2 / log_id, poses)
        gt_all += n_all
        gt_visible += n_vis
        stamps = [f["timestamp_ns"] for f in gt_frames]
        labels[log_id] = gt_frames
        preds[log_id] = load_pred(args.run / log_id, log_id, poses, stamps)
    n_pred_raw = sum(len(f["track_id"]) for f in av2_utils.ungroup_frames(preds))

    # Range filter, then drivable-area ROI, on both. av2's evaluate() copies
    # ego_translation_m from labels to predictions with annotate_frame_metadata, which
    # pairs frames by list position; filter_max_dist regroups frames in global timestamp
    # order, so with several logs that pairing hands predictions another log's ego pose.
    # Every frame here already carries its own ego pose, so the copy is skipped.
    labels = filter_max_dist(labels, MAX_RANGE_M)
    preds = filter_max_dist(preds, MAX_RANGE_M)
    check_alignment(labels, preds)
    if args.upstream_roi:
        labels = filter_drivable_area(labels, str(args.av2))
        preds = filter_drivable_area(preds, str(args.av2))
    else:
        labels = filter_roi_city(labels, args.av2)
        preds = filter_roi_city(preds, args.av2)
    n_gt_eval = sum(len(f["track_id"]) for f in av2_utils.ungroup_frames(labels))
    n_pred_eval = sum(len(f["track_id"]) for f in av2_utils.ungroup_frames(preds))

    out_dir = args.run / "trackeval"
    with contextlib.redirect_stdout(io.StringIO()):
        res = evaluate_tracking(labels, preds, [CLASS], tracker_name="TRACKER", output_dir=str(out_dir))
    combined = res["TrackEvalDataset"]["TRACKER"]["COMBINED_SEQ"][CLASS]
    h, c = combined["HOTA"], combined["CLEAR"]
    summary = {
        "run": str(args.run),
        "logs": len(log_ids),
        "roi_filter": "av2-0.3.6-upstream" if args.upstream_roi else "city-frame",
        "gt_boxes_all": gt_all,
        "gt_boxes_with_points": gt_visible,
        "gt_boxes_evaluated": n_gt_eval,
        "pred_boxes_raw": n_pred_raw,
        "pred_boxes_evaluated": n_pred_eval,
        "HOTA": float(np.mean(h["HOTA"])),
        "DetA": float(np.mean(h["DetA"])),
        "AssA": float(np.mean(h["AssA"])),
        "LocA": float(np.mean(h["LocA"])),
        "DetRe": float(np.mean(h["DetRe"])),
        "DetPr": float(np.mean(h["DetPr"])),
        "MOTA": float(c["MOTA"]),
        "MOTP": float(c["MOTP"]),
        "IDSW": int(c["IDSW"]),
        "CLR_TP": int(c["CLR_TP"]),
        "CLR_FN": int(c["CLR_FN"]),
        "CLR_FP": int(c["CLR_FP"]),
        "MT": int(c["MT"]),
        "ML": int(c["ML"]),
        "per_log_HOTA": {
            s: float(np.mean(v[CLASS]["HOTA"]["HOTA"]))
            for s, v in res["TrackEvalDataset"]["TRACKER"].items() if s != "COMBINED_SEQ"
        },
    }
    out = args.out or (args.run / "score.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_log_HOTA"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
