"""Detector -> tracker -> monitor as composable nodes in one process, intra-process comms on.

  ros2 launch lidar_tracking pipeline.launch.py                 # frozen config
  ros2 bag play data/bags/<log>                                  # in another terminal
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    default_params = str(Path(get_package_share_directory("lidar_tracking")) / "config" / "final.yaml")
    params = LaunchConfiguration("params")
    ipc = {"use_intra_process_comms": True}
    nodes = [
        ComposableNode(package="lidar_tracking", plugin=f"lidar_tracking::{p}", name=n,
                       parameters=[params], extra_arguments=[ipc])
        for p, n in [("DetectorComponent", "lidar_detector"),
                     ("TrackerComponent", "lidar_tracker"),
                     ("MonitorComponent", "track_monitor")]
    ]
    return LaunchDescription([
        DeclareLaunchArgument("params", default_value=default_params),
        ComposableNodeContainer(name="lidar_pipeline", namespace="", package="rclcpp_components",
                                executable="component_container", composable_node_descriptions=nodes,
                                output="screen"),
    ])
