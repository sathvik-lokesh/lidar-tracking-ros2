#include "lidar_tracking/ros_conversions.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <cmath>

namespace lidar_tracking
{

std::vector<Point> from_cloud2(const sensor_msgs::msg::PointCloud2 & msg)
{
  std::vector<Point> out;
  const size_t n = static_cast<size_t>(msg.width) * msg.height;
  out.reserve(n);
  bool has_intensity = false;
  for (const auto & f : msg.fields) {
    if (f.name == "intensity" && f.datatype == sensor_msgs::msg::PointField::FLOAT32) {has_intensity = true;}
  }
  sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
  if (has_intensity) {
    sensor_msgs::PointCloud2ConstIterator<float> it(msg, "intensity");
    for (size_t i = 0; i < n; ++i, ++x, ++y, ++z, ++it) {
      if (std::isfinite(*x) && std::isfinite(*y) && std::isfinite(*z)) {out.push_back({*x, *y, *z, *it});}
    }
  } else {
    for (size_t i = 0; i < n; ++i, ++x, ++y, ++z) {
      if (std::isfinite(*x) && std::isfinite(*y) && std::isfinite(*z)) {out.push_back({*x, *y, *z, 0.f});}
    }
  }
  return out;
}

vision_msgs::msg::Detection3DArray to_detection_array(
  const std::vector<Box> & boxes, const std_msgs::msg::Header & header,
  const std::vector<std::string> & ids)
{
  vision_msgs::msg::Detection3DArray arr;
  arr.header = header;
  arr.detections.reserve(boxes.size());
  for (size_t i = 0; i < boxes.size(); ++i) {
    const auto & b = boxes[i];
    vision_msgs::msg::Detection3D d;
    d.header = header;
    if (i < ids.size()) {d.id = ids[i];}
    auto & c = d.bbox.center;
    c.position.x = b.cx;
    c.position.y = b.cy;
    c.position.z = b.cz;
    c.orientation.w = std::cos(0.5 * b.yaw);
    c.orientation.z = std::sin(0.5 * b.yaw);
    d.bbox.size.x = b.length;
    d.bbox.size.y = b.width;
    d.bbox.size.z = b.height;
    vision_msgs::msg::ObjectHypothesisWithPose h;
    h.hypothesis.class_id = "object";
    h.hypothesis.score = 1.0;
    h.pose.pose = c;
    d.results.push_back(h);
    arr.detections.push_back(std::move(d));
  }
  return arr;
}

}  // namespace lidar_tracking
