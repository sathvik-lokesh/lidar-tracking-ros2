// The pipeline as three composable nodes (load them in one container with
// intra-process comms, see launch/pipeline.launch.py):
//
//   /av2/lidar --[DetectorComponent]--> /detections --[TrackerComponent]--> /tracks
//                                                                             |
//                                        /monitor/flags <--[MonitorComponent]--+
//
// Each node publishes its own processing-time stats on /diagnostics against a
// per-node budget (detector 80 ms + tracker 10 ms + monitor 5 ms < 100 ms frame period).
// Tracker and monitor need the ego pose: tf2 MessageFilters hold each message until
// map <- message-frame is available at its stamp.
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <message_filters/subscriber.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/create_timer_ros.hpp>
#include <tf2_ros/message_filter.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "lidar_tracking/detector.hpp"
#include "lidar_tracking/monitor.hpp"
#include "lidar_tracking/node_utils.hpp"
#include "lidar_tracking/ros_conversions.hpp"
#include "lidar_tracking/tracker.hpp"

namespace lidar_tracking
{

namespace
{
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
// Reliable, not SensorDataQoS: a ~1.9 MB cloud is hundreds of UDP fragments, and with
// best effort losing any one drops the message (measured: 1 of ~80 arrived).
rclcpp::QoS reliable_qos() {return rclcpp::QoS(rclcpp::KeepLast(10)).reliable();}

// tf buffer + listener + MessageFilter on `topic`, in one place.
template<class MsgT>
struct TfGated
{
  std::unique_ptr<tf2_ros::Buffer> buffer;
  std::unique_ptr<tf2_ros::TransformListener> listener;
  message_filters::Subscriber<MsgT> sub;
  std::unique_ptr<tf2_ros::MessageFilter<MsgT>> filter;

  void init(rclcpp::Node * node, const std::string & topic, const std::string & target_frame)
  {
    buffer = std::make_unique<tf2_ros::Buffer>(node->get_clock());
    buffer->setCreateTimerInterface(
      std::make_shared<tf2_ros::CreateTimerROS>(node->get_node_base_interface(), node->get_node_timers_interface()));
    listener = std::make_unique<tf2_ros::TransformListener>(*buffer);
    sub.subscribe(node, topic, reliable_qos());
    filter = std::make_unique<tf2_ros::MessageFilter<MsgT>>(
      sub, *buffer, target_frame, 20, node->get_node_logging_interface(), node->get_node_clock_interface(),
      std::chrono::seconds(1));
  }
};
}  // namespace

// ---------------------------------------------------------------- detector
class DetectorComponent : public rclcpp::Node
{
public:
  explicit DetectorComponent(const rclcpp::NodeOptions & options)
  : Node("lidar_detector", options), detector_(declare_detector_params(*this)),
    stats_(*this, "lidar_detector", declare_parameter("budget_ms", 80.0))
  {
    pub_ = create_publisher<vision_msgs::msg::Detection3DArray>("detections", reliable_qos());
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/av2/lidar", reliable_qos(),
      [this](sensor_msgs::msg::PointCloud2::UniquePtr msg) {
        const auto t0 = Clock::now();
        StageTimings t;
        const auto boxes = detector_.detect(from_cloud2(*msg), &t);
        auto out = std::make_unique<vision_msgs::msg::Detection3DArray>(to_detection_array(boxes, msg->header));
        pub_->publish(std::move(out));
        stats_.add(ms_since(t0));
        RCLCPP_DEBUG(get_logger(), "boxes=%d detect %.1f ms", t.n_boxes, t.total_ms);
      });
  }

private:
  Detector detector_;
  LatencyStats stats_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

// ---------------------------------------------------------------- tracker
class TrackerComponent : public rclcpp::Node
{
public:
  explicit TrackerComponent(const rclcpp::NodeOptions & options)
  : Node("lidar_tracker", options), tracker_(declare_tracker_params(*this)),
    stats_(*this, "lidar_tracker", declare_parameter("budget_ms", 10.0))
  {
    map_frame_ = declare_parameter("map_frame", std::string("map"));
    pub_ = create_publisher<vision_msgs::msg::Detection3DArray>("tracks", reliable_qos());
    tf_.init(this, "detections", map_frame_);
    tf_.filter->registerCallback(&TrackerComponent::on_detections, this);
  }

private:
  void on_detections(const vision_msgs::msg::Detection3DArray::ConstSharedPtr & msg)
  {
    const auto t0 = Clock::now();
    Pose2D ego;
    try {
      ego = pose_from_tf(tf_.buffer->lookupTransform(map_frame_, msg->header.frame_id, msg->header.stamp));
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN(get_logger(), "no pose: %s", e.what());
      return;
    }
    const auto tracks = tracker_.update(from_detection_array(*msg), ego, rclcpp::Time(msg->header.stamp).seconds());
    std::vector<Box> boxes;
    std::vector<std::string> ids;
    for (const auto & o : tracks) {
      boxes.push_back(o.box);
      ids.push_back(std::to_string(o.id));
    }
    pub_->publish(to_detection_array(boxes, msg->header, ids));
    stats_.add(ms_since(t0));
    RCLCPP_DEBUG(get_logger(), "tracks=%zu", tracks.size());
  }

  Tracker tracker_;
  LatencyStats stats_;
  std::string map_frame_;
  TfGated<vision_msgs::msg::Detection3DArray> tf_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr pub_;
};

// ---------------------------------------------------------------- monitor
class MonitorComponent : public rclcpp::Node
{
public:
  explicit MonitorComponent(const rclcpp::NodeOptions & options)
  : Node("track_monitor", options), monitor_(declare_monitor_params(*this)),
    stats_(*this, "track_monitor", declare_parameter("budget_ms", 5.0))
  {
    map_frame_ = declare_parameter("map_frame", std::string("map"));
    flags_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>("monitor/flags", reliable_qos());
    diag_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
    tf_.init(this, "tracks", map_frame_);
    tf_.filter->registerCallback(&MonitorComponent::on_tracks, this);
  }

private:
  void on_tracks(const vision_msgs::msg::Detection3DArray::ConstSharedPtr & msg)
  {
    const auto t0 = Clock::now();
    Pose2D ego;
    try {
      ego = pose_from_tf(tf_.buffer->lookupTransform(map_frame_, msg->header.frame_id, msg->header.stamp));
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN(get_logger(), "no pose: %s", e.what());
      return;
    }
    const auto boxes = from_detection_array(*msg);
    std::vector<TrackOutput> reports;
    for (size_t i = 0; i < boxes.size(); ++i) {
      reports.push_back({boxes[i], std::stoll(msg->detections[i].id), 0});
    }
    const auto flags = monitor_.check(reports, ego, rclcpp::Time(msg->header.stamp).seconds());

    // Flagged tracks, with the flag mask in the hypothesis class_id ("speed|size" ...).
    vision_msgs::msg::Detection3DArray flagged;
    flagged.header = msg->header;
    for (size_t i = 0; i < flags.size(); ++i) {
      if (!flags[i]) {continue;}
      auto d = msg->detections[i];
      std::string why;
      if (flags[i] & kFlagSpeed) {why += "speed|";}
      if (flags[i] & kFlagSmallSpeed) {why += "small_speed|";}
      if (flags[i] & kFlagSize) {why += "size|";}
      why.pop_back();
      if (!d.results.empty()) {d.results[0].hypothesis.class_id = why;}
      flagged.detections.push_back(d);
    }
    flags_pub_->publish(flagged);
    if (!flagged.detections.empty()) {
      RCLCPP_INFO(get_logger(), "%zu of %zu tracks flagged", flagged.detections.size(), flags.size());
    }
    if (++frames_ % 50 == 0) {publish_counts();}
    stats_.add(ms_since(t0));
  }

  void publish_counts()
  {
    const auto & c = monitor_.counts();
    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = now();
    diagnostic_msgs::msg::DiagnosticStatus st;
    st.name = "track_monitor: flags";
    st.hardware_id = get_fully_qualified_name();
    st.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    st.message = "cumulative counts";
    auto kv = [&](const char * k, int64_t v) {
        diagnostic_msgs::msg::KeyValue x; x.key = k; x.value = std::to_string(v); st.values.push_back(x);
      };
    kv("reports", c.reports);
    kv("flag_any", c.any);
    kv("flag_speed", c.speed);
    kv("flag_small_speed", c.small_speed);
    kv("flag_size", c.size);
    arr.status.push_back(st);
    diag_pub_->publish(arr);
  }

  TrackMonitor monitor_;
  LatencyStats stats_;
  std::string map_frame_;
  int64_t frames_{0};
  TfGated<vision_msgs::msg::Detection3DArray> tf_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr flags_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
};

}  // namespace lidar_tracking

RCLCPP_COMPONENTS_REGISTER_NODE(lidar_tracking::DetectorComponent)
RCLCPP_COMPONENTS_REGISTER_NODE(lidar_tracking::TrackerComponent)
RCLCPP_COMPONENTS_REGISTER_NODE(lidar_tracking::MonitorComponent)
