#!/usr/bin/env python3
"""Record what the live pipeline publishes, for comparison with process_bag.

Writes JSON: per-stamp track counts and ids on /tracks, number of flagged tracks
on /monitor/flags, and the last /diagnostics status per name.

  source env.sh && python3 -P scripts/listen_pipeline.py --out /tmp/x.json --seconds 25
"""
from __future__ import annotations

import argparse
import json
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection3DArray


def stamp_ns(h) -> int:
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=25.0)
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("pipeline_listener")
    qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
    tracks: dict[int, list[str]] = {}
    flags: dict[int, int] = {}
    diags: dict[str, dict] = {}
    node.create_subscription(Detection3DArray, "/tracks",
                             lambda m: tracks.__setitem__(stamp_ns(m.header), [d.id for d in m.detections]), qos)
    node.create_subscription(Detection3DArray, "/monitor/flags",
                             lambda m: flags.__setitem__(stamp_ns(m.header), len(m.detections)), qos)

    def on_diag(m):
        for s in m.status:
            diags[s.name] = {"level": int.from_bytes(s.level, "little") if isinstance(s.level, bytes) else int(s.level),
                             "message": s.message, **{kv.key: kv.value for kv in s.values}}
    node.create_subscription(DiagnosticArray, "/diagnostics", on_diag, 50)

    end = time.time() + args.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    json.dump({"tracks": {str(k): v for k, v in sorted(tracks.items())},
               "flags": {str(k): v for k, v in sorted(flags.items())}, "diagnostics": diags},
              open(args.out, "w"), indent=1)
    print(f"tracks frames {len(tracks)}, flag frames {len(flags)}, diagnostics {len(diags)}")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
