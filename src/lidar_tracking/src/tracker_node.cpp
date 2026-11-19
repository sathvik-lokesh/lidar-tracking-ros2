// Live node: /av2/lidar (PointCloud2) -> /detections and /tracks (Detection3DArray,
// same frame as the cloud). /tracks carries the track id in Detection3D::id.
//
// The tracker runs in `map_frame`, so each cloud waits (tf2_ros::MessageFilter) until
// the map <- cloud-frame transform at its stamp is available. Parameters mirror
// DetectorParams and TrackerParams; config/final.yaml holds the frozen settings.
#include <message_filters/subscriber.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/create_timer_ros.hpp>
#include <tf2_ros/message_filter.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "lidar_tracking/detector.hpp"
#include "lidar_tracking/ros_conversions.hpp"
#include "lidar_tracking/tracker.hpp"

namespace lidar_tracking
{

class TrackerNode : public rclcpp::Node
{
public:
  explicit TrackerNode(const rclcpp::NodeOptions & options)
  : Node("lidar_tracker", options)
  {
    DetectorParams p;
    p.max_range_m = declare_parameter("max_range_m", p.max_range_m);
    p.ground_cell_m = declare_parameter("ground_cell_m", p.ground_cell_m);
    p.ground_neighbourhood = static_cast<int>(declare_parameter("ground_neighbourhood", static_cast<int64_t>(p.ground_neighbourhood)));
    p.ground_clearance_m = declare_parameter("ground_clearance_m", p.ground_clearance_m);
    p.max_height_m = declare_parameter("max_height_m", p.max_height_m);
    p.voxel_m = declare_parameter("voxel_m", p.voxel_m);
    p.cluster_2d = declare_parameter("cluster_2d", p.cluster_2d);
    p.cluster_method = static_cast<int>(declare_parameter("cluster_method", static_cast<int64_t>(p.cluster_method)));
    p.cluster_tolerance_m = declare_parameter("cluster_tolerance_m", p.cluster_tolerance_m);
    p.min_cluster_points = static_cast<int>(declare_parameter("min_cluster_points", static_cast<int64_t>(p.min_cluster_points)));
    p.max_cluster_points = static_cast<int>(declare_parameter("max_cluster_points", static_cast<int64_t>(p.max_cluster_points)));
    p.max_box_length_m = declare_parameter("max_box_length_m", p.max_box_length_m);
    p.max_box_area_m2 = declare_parameter("max_box_area_m2", p.max_box_area_m2);
    p.min_box_height_m = declare_parameter("min_box_height_m", p.min_box_height_m);
    p.min_box_width_m = declare_parameter("min_box_width_m", p.min_box_width_m);
    p.max_clearance_m = declare_parameter("max_clearance_m", p.max_clearance_m);
    detector_ = std::make_unique<Detector>(p);

    TrackerParams tp;
    const std::string mode = declare_parameter("tracker", std::string("kf"));
    tp.mode = mode == "none" ? TrackerParams::Mode::None :
      mode == "nn" ? TrackerParams::Mode::Nearest : TrackerParams::Mode::Kalman;
    tp.gate_m = declare_parameter("gate_m", tp.gate_m);
    tp.min_hits = static_cast<int>(declare_parameter("min_hits", static_cast<int64_t>(tp.min_hits)));
    tp.max_misses = static_cast<int>(declare_parameter("max_misses", static_cast<int64_t>(tp.max_misses)));
    tp.accel_std = declare_parameter("accel_std", tp.accel_std);
    tp.meas_std_m = declare_parameter("meas_std_m", tp.meas_std_m);
    tp.output_filtered = declare_parameter("output_filtered", tp.output_filtered);
    tracker_ = std::make_unique<Tracker>(tp);
    map_frame_ = declare_parameter("map_frame", std::string("map"));

    det_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>("detections", 10);
    trk_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>("tracks", 10);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_buffer_->setCreateTimerInterface(
      std::make_shared<tf2_ros::CreateTimerROS>(get_node_base_interface(), get_node_timers_interface()));
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // Reliable, not SensorDataQoS: a ~1.9 MB cloud is hundreds of UDP fragments, and
    // with best effort losing any one drops the message (measured: 1 of ~80 arrived).
    cloud_sub_.subscribe(this, "/av2/lidar", rclcpp::QoS(rclcpp::KeepLast(5)).reliable());
    tf_filter_ = std::make_unique<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>>(
      cloud_sub_, *tf_buffer_, map_frame_, 10, get_node_logging_interface(), get_node_clock_interface(),
      std::chrono::seconds(1));
    tf_filter_->registerCallback(&TrackerNode::on_cloud, this);
  }

private:
  void on_cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
  {
    Pose2D ego;
    try {
      const auto tf = tf_buffer_->lookupTransform(map_frame_, msg->header.frame_id, msg->header.stamp);
      const auto & q = tf.transform.rotation;
      ego.x = tf.transform.translation.x;
      ego.y = tf.transform.translation.y;
      ego.yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN(get_logger(), "no %s pose for cloud: %s", map_frame_.c_str(), e.what());
      return;
    }
    StageTimings t;
    const auto boxes = detector_->detect(from_cloud2(*msg), &t);
    const auto t0 = std::chrono::steady_clock::now();
    const double stamp_s = rclcpp::Time(msg->header.stamp).seconds();
    const auto tracks = tracker_->update(boxes, ego, stamp_s);
    const double track_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    det_pub_->publish(to_detection_array(boxes, msg->header));
    std::vector<Box> tb;
    std::vector<std::string> ids;
    tb.reserve(tracks.size());
    ids.reserve(tracks.size());
    for (const auto & o : tracks) {
      tb.push_back(o.box);
      ids.push_back(std::to_string(o.id));
    }
    trk_pub_->publish(to_detection_array(tb, msg->header, ids));
    RCLCPP_DEBUG(
      get_logger(), "in=%d above=%d clusters=%d boxes=%d tracks=%zu | ground %.1f cluster %.1f fit %.1f detect %.1f track %.1f ms",
      t.n_in, t.n_nonground, t.n_clusters, t.n_boxes, tracks.size(), t.ground_ms, t.cluster_ms, t.fit_ms, t.total_ms, track_ms);
  }

  std::unique_ptr<Detector> detector_;
  std::unique_ptr<Tracker> tracker_;
  std::string map_frame_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr det_pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr trk_pub_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
  std::unique_ptr<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>> tf_filter_;
};

}  // namespace lidar_tracking

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(lidar_tracking::TrackerNode)
