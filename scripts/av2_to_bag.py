#!/usr/bin/env python3
"""Convert Argoverse 2 sensor logs (lidar only) into rosbag2 (mcap) files.

One bag per log, written to <out>/<log_id>/:
  /av2/lidar  sensor_msgs/PointCloud2     frame base_link (AV2 sweeps are already in the ego frame)
              fields x, y, z, intensity (float32), ring (uint16); both VLP-32s merged, 10 Hz
  /tf         tf2_msgs/TFMessage           map -> base_link, from city_SE3_egovehicle
  /av2/gt     vision_msgs/Detection3DArray ground-truth cuboids in base_link (id = track uuid,
              class_id = AV2 category). For viewing only; scoring reads annotations.feather.

Every message is stamped with the sweep timestamp, and the bag receive time equals it,
so `ros2 bag play` replays at the recorded 10 Hz.

After writing, each bag is read back and checked: message counts, stamps, and that the
first and last clouds match the source feather bit for bit (float16 -> float32 is exact).

Run with the ROS env sourced:
  source env.sh && .venv/bin/python -P scripts/av2_to_bag.py --av2 data/av2/val --out data/bags --n-logs 1
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.feather as pf
import rosbag2_py
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import PointCloud2, PointField
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

CLOUD_DTYPE = np.dtype(
    {
        "names": ["x", "y", "z", "intensity", "ring"],
        "formats": ["<f4", "<f4", "<f4", "<f4", "<u2"],
        "offsets": [0, 4, 8, 12, 16],
        "itemsize": 20,  # 2 bytes padding keeps every point 4-byte aligned
    }
)
CLOUD_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="ring", offset=16, datatype=PointField.UINT16, count=1),
]
TOPICS = {
    "/av2/lidar": "sensor_msgs/msg/PointCloud2",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/av2/gt": "vision_msgs/msg/Detection3DArray",
}
MAX_POSE_GAP_NS = 5_000_000  # a sweep must have a pose within 5 ms


def stamp(ns: int) -> Time:
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def load_sweep(path: Path) -> np.ndarray:
    t = pf.read_table(path, columns=["x", "y", "z", "intensity", "laser_number"])
    pts = np.zeros(t.num_rows, dtype=CLOUD_DTYPE)  # zeros: pad bytes must be deterministic
    for name in ("x", "y", "z"):
        pts[name] = t[name].to_numpy().astype(np.float32)
    pts["intensity"] = t["intensity"].to_numpy().astype(np.float32)
    pts["ring"] = t["laser_number"].to_numpy().astype(np.uint16)
    return pts


def cloud_msg(pts: np.ndarray, ns: int) -> PointCloud2:
    msg = PointCloud2()
    msg.header.stamp = stamp(ns)
    msg.header.frame_id = "base_link"
    msg.height = 1
    msg.width = len(pts)
    msg.fields = CLOUD_FIELDS
    msg.is_bigendian = False
    msg.point_step = CLOUD_DTYPE.itemsize
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = pts.tobytes()
    return msg


def cloud_from_msg(msg: PointCloud2) -> np.ndarray:
    return np.frombuffer(bytes(msg.data), dtype=CLOUD_DTYPE, count=msg.width)


def tf_msg(pose: dict, ns: int) -> TFMessage:
    tf = TransformStamped()
    tf.header.stamp = stamp(ns)
    tf.header.frame_id = "map"
    tf.child_frame_id = "base_link"
    tf.transform.translation.x = pose["tx_m"]
    tf.transform.translation.y = pose["ty_m"]
    tf.transform.translation.z = pose["tz_m"]
    tf.transform.rotation.w = pose["qw"]
    tf.transform.rotation.x = pose["qx"]
    tf.transform.rotation.y = pose["qy"]
    tf.transform.rotation.z = pose["qz"]
    return TFMessage(transforms=[tf])


def gt_msg(rows: list[dict], ns: int) -> Detection3DArray:
    arr = Detection3DArray()
    arr.header.stamp = stamp(ns)
    arr.header.frame_id = "base_link"
    for r in rows:
        d = Detection3D()
        d.header = arr.header
        d.id = r["track_uuid"]
        c = d.bbox.center
        c.position.x, c.position.y, c.position.z = r["tx_m"], r["ty_m"], r["tz_m"]
        c.orientation.w, c.orientation.x = r["qw"], r["qx"]
        c.orientation.y, c.orientation.z = r["qy"], r["qz"]
        d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = r["length_m"], r["width_m"], r["height_m"]
        h = ObjectHypothesisWithPose()
        h.hypothesis.class_id = r["category"]
        h.hypothesis.score = 1.0
        h.pose.pose = c
        d.results = [h]
        arr.detections.append(d)
    return arr


def group_by_ts(table) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in table.to_pylist():
        out.setdefault(r["timestamp_ns"], []).append(r)
    return out


def convert_log(log_dir: Path, bag_dir: Path) -> dict:
    sweeps = sorted((log_dir / "sensors" / "lidar").glob("*.feather"))
    sweep_ns = [int(p.stem) for p in sweeps]
    poses = pf.read_table(log_dir / "city_SE3_egovehicle.feather")
    pose_ns = poses["timestamp_ns"].to_numpy()
    pose_rows = poses.to_pylist()
    gt = group_by_ts(pf.read_table(log_dir / "annotations.feather"))

    if bag_dir.exists():
        shutil.rmtree(bag_dir)
    bag_dir.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap", storage_preset_profile="zstd_fast"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for i, (name, typ) in enumerate(TOPICS.items()):
        writer.create_topic(rosbag2_py.TopicMetadata(id=i, name=name, type=typ, serialization_format="cdr"))

    n_points = n_boxes = 0
    worst_gap = 0
    for path, ns in zip(sweeps, sweep_ns):
        j = int(np.argmin(np.abs(pose_ns - ns)))
        gap = abs(int(pose_ns[j]) - ns)
        if gap > MAX_POSE_GAP_NS:
            raise RuntimeError(f"{log_dir.name}: no pose within 5 ms of sweep {ns} (gap {gap} ns)")
        worst_gap = max(worst_gap, gap)
        pts = load_sweep(path)
        boxes = gt.get(ns, [])
        writer.write("/tf", serialize_message(tf_msg(pose_rows[j], ns)), ns)
        writer.write("/av2/lidar", serialize_message(cloud_msg(pts, ns)), ns)
        writer.write("/av2/gt", serialize_message(gt_msg(boxes, ns)), ns)
        n_points += len(pts)
        n_boxes += len(boxes)
    del writer  # closes the bag

    unmatched_gt = sorted(set(gt) - set(sweep_ns))
    return {
        "log": log_dir.name,
        "sweeps": len(sweeps),
        "points": n_points,
        "boxes": n_boxes,
        "gt_stamps_without_sweep": len(unmatched_gt),
        "worst_pose_gap_ms": worst_gap / 1e6,
        "first": (sweeps[0], sweep_ns[0]),
        "last": (sweeps[-1], sweep_ns[-1]),
    }


def verify_bag(bag_dir: Path, info: dict) -> list[str]:
    """Read the bag back. Returns a list of failures (empty = pass)."""
    fails: list[str] = []
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    counts = {t: 0 for t in TOPICS}
    clouds = {}
    prev_ns = -1
    n_boxes = 0
    while reader.has_next():
        topic, data, t_ns = reader.read_next_ext()[:3] if hasattr(reader, "read_next_ext") else reader.read_next()
        counts[topic] += 1
        if t_ns < prev_ns:
            fails.append(f"receive time goes backwards at {t_ns}")
        prev_ns = t_ns
        if topic == "/av2/lidar":
            msg = deserialize_message(data, PointCloud2)
            h_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            if h_ns != t_ns:
                fails.append(f"cloud header stamp {h_ns} != receive time {t_ns}")
            if t_ns in (info["first"][1], info["last"][1]):
                clouds[t_ns] = bytes(msg.data)  # raw bytes: a structured .copy() may drop pad bytes
        elif topic == "/av2/gt":
            n_boxes += len(deserialize_message(data, Detection3DArray).detections)
    del reader

    for t in TOPICS:
        if counts[t] != info["sweeps"]:
            fails.append(f"{t}: {counts[t]} msgs, expected {info['sweeps']}")
    if n_boxes != info["boxes"]:
        fails.append(f"gt boxes {n_boxes} != {info['boxes']}")
    for path, ns in (info["first"], info["last"]):
        src = load_sweep(path)
        raw = clouds.get(ns)
        got = None if raw is None else np.frombuffer(raw, dtype=CLOUD_DTYPE)
        same = got is not None and len(got) == len(src) and all(np.array_equal(got[f], src[f]) for f in CLOUD_DTYPE.names)
        if not same or raw != src.tobytes():
            fails.append(f"cloud at {ns} does not match {path.name}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--av2", type=Path, default=Path("data/av2/val"), help="AV2 split dir with one folder per log")
    ap.add_argument("--out", type=Path, default=Path("data/bags"))
    ap.add_argument("--n-logs", type=int, default=None, help="first N logs in sorted order (default: all)")
    ap.add_argument("--log", action="append", default=[], help="convert only this log id (repeatable)")
    ap.add_argument("--skip-existing", action="store_true", help="skip logs whose bag already exists")
    args = ap.parse_args()

    logs = sorted(p for p in args.av2.iterdir() if (p / "sensors" / "lidar").is_dir())
    if args.log:
        logs = [p for p in logs if p.name in set(args.log)]
    if args.n_logs is not None:
        logs = logs[: args.n_logs]
    if not logs:
        print("no logs found", file=sys.stderr)
        return 1

    n_fail = 0
    for k, log_dir in enumerate(logs, 1):
        bag_dir = args.out / log_dir.name
        if args.skip_existing and (bag_dir / "metadata.yaml").exists():
            print(f"[{k}/{len(logs)}] {log_dir.name} exists, skipped")
            continue
        t0 = time.time()
        info = convert_log(log_dir, bag_dir)
        fails = verify_bag(bag_dir, info)
        size_mb = sum(f.stat().st_size for f in bag_dir.iterdir()) / 1e6
        status = "OK" if not fails else "FAIL"
        print(
            f"[{k}/{len(logs)}] {log_dir.name} {status}: {info['sweeps']} sweeps, "
            f"{info['points'] / info['sweeps']:.0f} pts/sweep, {info['boxes']} gt boxes, "
            f"gt stamps w/o sweep {info['gt_stamps_without_sweep']}, worst pose gap "
            f"{info['worst_pose_gap_ms']:.2f} ms, {size_mb:.0f} MB, {time.time() - t0:.1f} s"
        )
        for f in fails:
            print(f"    FAIL {f}")
        n_fail += bool(fails)
    print(f"done: {len(logs) - n_fail}/{len(logs)} logs verified")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
