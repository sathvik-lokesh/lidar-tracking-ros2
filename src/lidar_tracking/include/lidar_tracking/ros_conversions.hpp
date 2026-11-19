// ROS message <-> core type conversions (no PCL).
#pragma once

#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>

#include <string>
#include <vector>

#include "lidar_tracking/types.hpp"

namespace lidar_tracking
{

// Reads x, y, z (float32) and intensity if present. Non-finite points are skipped.
std::vector<Point> from_cloud2(const sensor_msgs::msg::PointCloud2 & msg);

// One Detection3D per box. ids[i] (if given) goes into Detection3D::id.
vision_msgs::msg::Detection3DArray to_detection_array(
  const std::vector<Box> & boxes, const std_msgs::msg::Header & header,
  const std::vector<std::string> & ids = {});

}  // namespace lidar_tracking
