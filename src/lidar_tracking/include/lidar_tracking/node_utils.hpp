// Helpers shared by the ROS components: parameter declaration and latency stats.
#pragma once

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "lidar_tracking/monitor.hpp"
#include "lidar_tracking/tracker.hpp"
#include "lidar_tracking/types.hpp"

namespace lidar_tracking
{

inline DetectorParams declare_detector_params(rclcpp::Node & n)
{
  DetectorParams p;
  auto i = [&](const char * k, int v) {return static_cast<int>(n.declare_parameter(k, static_cast<int64_t>(v)));};
  p.max_range_m = n.declare_parameter("max_range_m", p.max_range_m);
  p.ground_cell_m = n.declare_parameter("ground_cell_m", p.ground_cell_m);
  p.ground_neighbourhood = i("ground_neighbourhood", p.ground_neighbourhood);
  p.ground_clearance_m = n.declare_parameter("ground_clearance_m", p.ground_clearance_m);
  p.max_height_m = n.declare_parameter("max_height_m", p.max_height_m);
  p.voxel_m = n.declare_parameter("voxel_m", p.voxel_m);
  p.cluster_2d = n.declare_parameter("cluster_2d", p.cluster_2d);
  p.cluster_method = i("cluster_method", p.cluster_method);
  p.cluster_tolerance_m = n.declare_parameter("cluster_tolerance_m", p.cluster_tolerance_m);
  p.min_cluster_points = i("min_cluster_points", p.min_cluster_points);
  p.max_cluster_points = i("max_cluster_points", p.max_cluster_points);
  p.max_box_length_m = n.declare_parameter("max_box_length_m", p.max_box_length_m);
  p.max_box_area_m2 = n.declare_parameter("max_box_area_m2", p.max_box_area_m2);
  p.min_box_height_m = n.declare_parameter("min_box_height_m", p.min_box_height_m);
  p.min_box_width_m = n.declare_parameter("min_box_width_m", p.min_box_width_m);
  p.max_clearance_m = n.declare_parameter("max_clearance_m", p.max_clearance_m);
  return p;
}

inline TrackerParams declare_tracker_params(rclcpp::Node & n)
{
  TrackerParams p;
  const std::string mode = n.declare_parameter("tracker", std::string("kf"));
  p.mode = mode == "none" ? TrackerParams::Mode::None :
    mode == "nn" ? TrackerParams::Mode::Nearest : TrackerParams::Mode::Kalman;
  p.gate_m = n.declare_parameter("gate_m", p.gate_m);
  p.min_hits = static_cast<int>(n.declare_parameter("min_hits", static_cast<int64_t>(p.min_hits)));
  p.max_misses = static_cast<int>(n.declare_parameter("max_misses", static_cast<int64_t>(p.max_misses)));
  p.accel_std = n.declare_parameter("accel_std", p.accel_std);
  p.meas_std_m = n.declare_parameter("meas_std_m", p.meas_std_m);
  p.output_filtered = n.declare_parameter("output_filtered", p.output_filtered);
  return p;
}

inline MonitorParams declare_monitor_params(rclcpp::Node & n)
{
  MonitorParams p;
  p.max_speed_mps = n.declare_parameter("max_speed_mps", p.max_speed_mps);
  p.small_area_m2 = n.declare_parameter("small_area_m2", p.small_area_m2);
  p.max_small_speed_mps = n.declare_parameter("max_small_speed_mps", p.max_small_speed_mps);
  p.max_area_ratio = n.declare_parameter("max_area_ratio", p.max_area_ratio);
  return p;
}

inline Pose2D pose_from_tf(const geometry_msgs::msg::TransformStamped & tf)
{
  const auto & q = tf.transform.rotation;
  return {tf.transform.translation.x, tf.transform.translation.y,
    std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))};
}

// Rolling per-node processing time, published as a DiagnosticStatus every `window` frames.
class LatencyStats
{
public:
  LatencyStats(rclcpp::Node & node, std::string name, double budget_ms, size_t window = 50)
  : name_(std::move(name)), hw_id_(node.get_fully_qualified_name()), budget_ms_(budget_ms), window_(window)
  {
    pub_ = node.create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
    clock_ = node.get_clock();
  }

  void add(double ms)
  {
    ms_.push_back(ms);
    if (ms_.size() < window_) {return;}
    std::vector<double> s = ms_;
    std::sort(s.begin(), s.end());
    double mean = 0.0;
    for (double v : s) {mean += v;}
    mean /= static_cast<double>(s.size());
    const double p95 = s[static_cast<size_t>(0.95 * (s.size() - 1))];
    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = clock_->now();
    diagnostic_msgs::msg::DiagnosticStatus st;
    st.name = name_ + ": processing time";
    st.hardware_id = hw_id_;
    st.level = s.back() > budget_ms_ ? diagnostic_msgs::msg::DiagnosticStatus::WARN : diagnostic_msgs::msg::DiagnosticStatus::OK;
    st.message = s.back() > budget_ms_ ? "frame over budget" : "within budget";
    auto kv = [&](const char * k, double v) {
        diagnostic_msgs::msg::KeyValue x; x.key = k; x.value = std::to_string(v); st.values.push_back(x);
      };
    kv("frames", static_cast<double>(s.size()));
    kv("mean_ms", mean);
    kv("p95_ms", p95);
    kv("max_ms", s.back());
    kv("budget_ms", budget_ms_);
    arr.status.push_back(st);
    pub_->publish(arr);
    ms_.clear();
  }

private:
  std::string name_, hw_id_;
  double budget_ms_;
  size_t window_;
  std::vector<double> ms_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub_;
  rclcpp::Clock::SharedPtr clock_;
};

}  // namespace lidar_tracking
