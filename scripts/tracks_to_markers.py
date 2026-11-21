#!/usr/bin/env python3
"""Turn /tracks (and optionally /av2/gt) into a MarkerArray RViz can draw.

RViz has no built-in display for vision_msgs/Detection3DArray, so this is a
viewer helper only; nothing in the scored pipeline depends on it.

  source env.sh && python3 -P scripts/tracks_to_markers.py
  rviz2 -d rviz/pipeline.rviz
"""
from __future__ import annotations

import colorsys
import zlib

import rclpy
import rclpy.executors
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

LIFETIME = Duration(seconds=0.15).to_msg()  # a bit over one 10 Hz frame


def id_colour(track_id: str) -> tuple[float, float, float]:
    """Stable, well-spread colour per track id."""
    h = (zlib.crc32(track_id.encode()) % 360) / 360.0
    return colorsys.hsv_to_rgb(h, 0.85, 1.0)


class TracksToMarkers(Node):
    def __init__(self):
        super().__init__("tracks_to_markers")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(MarkerArray, "tracks/markers", 10)
        self.gt_pub = self.create_publisher(MarkerArray, "av2/gt/markers", 10)
        self.create_subscription(Detection3DArray, "tracks", self.on_tracks, qos)
        self.create_subscription(Detection3DArray, "av2/gt", self.on_gt, qos)

    def on_tracks(self, msg: Detection3DArray):
        out = MarkerArray()
        for i, d in enumerate(msg.detections):
            r, g, b = id_colour(d.id)
            box = Marker(header=msg.header, ns="tracks", id=2 * i, type=Marker.CUBE,
                         action=Marker.ADD, lifetime=LIFETIME)
            box.pose = d.bbox.center
            box.scale.x, box.scale.y, box.scale.z = d.bbox.size.x, d.bbox.size.y, max(d.bbox.size.z, 0.3)
            box.color.r, box.color.g, box.color.b, box.color.a = r, g, b, 0.55
            txt = Marker(header=msg.header, ns="ids", id=2 * i + 1, type=Marker.TEXT_VIEW_FACING,
                         action=Marker.ADD, lifetime=LIFETIME, text=d.id)
            txt.pose.position.x = d.bbox.center.position.x
            txt.pose.position.y = d.bbox.center.position.y
            txt.pose.position.z = d.bbox.center.position.z + box.scale.z / 2 + 0.8
            txt.scale.z = 1.2
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            out.markers += [box, txt]
        self.pub.publish(out)

    def on_gt(self, msg: Detection3DArray):
        out = MarkerArray()
        for i, d in enumerate(msg.detections):
            m = Marker(header=msg.header, ns="gt", id=i, type=Marker.LINE_LIST,
                       action=Marker.ADD, lifetime=LIFETIME)
            m.pose = d.bbox.center
            m.scale.x = 0.06
            m.color.g = m.color.a = 1.0
            m.color.r = m.color.b = 0.6
            hx, hy, hz = d.bbox.size.x / 2, d.bbox.size.y / 2, d.bbox.size.z / 2
            corners = [(sx * hx, sy * hy, sz * hz) for sz in (-1, 1) for sx, sy in
                       ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]
            for a, b in edges:
                m.points += [Point(x=corners[a][0], y=corners[a][1], z=corners[a][2]),
                             Point(x=corners[b][0], y=corners[b][1], z=corners[b][2])]
            out.markers.append(m)
        self.gt_pub.publish(out)


def main() -> int:
    rclpy.init()
    node = TracksToMarkers()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
