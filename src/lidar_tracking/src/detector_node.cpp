// Live node: /av2/lidar (PointCloud2) -> /detections (Detection3DArray), same frame.
// Parameters mirror DetectorParams. Per-frame stage timings go to the debug log.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>

#include <memory>

#include "lidar_tracking/detector.hpp"
#include "lidar_tracking/ros_conversions.hpp"

namespace lidar_tracking
{

class DetectorNode : public rclcpp::Node
{
public:
  explicit DetectorNode(const rclcpp::NodeOptions & options)
  : Node("lidar_detector", options)
  {
    DetectorParams p;
    p.max_range_m = declare_parameter("max_range_m", p.max_range_m);
    p.ground_cell_m = declare_parameter("ground_cell_m", p.ground_cell_m);
    p.ground_neighbourhood = static_cast<int>(declare_parameter("ground_neighbourhood", static_cast<int64_t>(p.ground_neighbourhood)));
    p.ground_clearance_m = declare_parameter("ground_clearance_m", p.ground_clearance_m);
    p.max_height_m = declare_parameter("max_height_m", p.max_height_m);
    p.voxel_m = declare_parameter("voxel_m", p.voxel_m);
    p.cluster_2d = declare_parameter("cluster_2d", p.cluster_2d);
    p.cluster_tolerance_m = declare_parameter("cluster_tolerance_m", p.cluster_tolerance_m);
    p.min_cluster_points = static_cast<int>(declare_parameter("min_cluster_points", static_cast<int64_t>(p.min_cluster_points)));
    p.max_cluster_points = static_cast<int>(declare_parameter("max_cluster_points", static_cast<int64_t>(p.max_cluster_points)));
    p.max_box_length_m = declare_parameter("max_box_length_m", p.max_box_length_m);
    p.max_box_area_m2 = declare_parameter("max_box_area_m2", p.max_box_area_m2);
    detector_ = std::make_unique<Detector>(p);

    pub_ = create_publisher<vision_msgs::msg::Detection3DArray>("detections", 10);
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      // Reliable, not SensorDataQoS: a ~1.9 MB cloud is hundreds of UDP fragments, and
      // with best effort losing any one drops the message (measured: 1 of ~80 arrived).
      "/av2/lidar", rclcpp::QoS(rclcpp::KeepLast(5)).reliable(),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {on_cloud(*msg);});
  }

private:
  void on_cloud(const sensor_msgs::msg::PointCloud2 & msg)
  {
    StageTimings t;
    const auto boxes = detector_->detect(from_cloud2(msg), &t);
    pub_->publish(to_detection_array(boxes, msg.header));
    RCLCPP_DEBUG(
      get_logger(), "in=%d above=%d voxel=%d clusters=%d boxes=%d | ground %.1f voxel %.1f cluster %.1f fit %.1f total %.1f ms",
      t.n_in, t.n_nonground, t.n_voxel, t.n_clusters, t.n_boxes, t.ground_ms, t.voxel_ms, t.cluster_ms, t.fit_ms, t.total_ms);
  }

  std::unique_ptr<Detector> detector_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

}  // namespace lidar_tracking

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(lidar_tracking::DetectorNode)
