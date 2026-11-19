// Offline runner: reads every cloud from a bag in order (no dropped frames, no
// wall-clock dependence), runs detector + tracker, and writes CSVs for scoring.
//
//   process_bag <bag_dir> <out_dir> [key=value ...]
//
// key=value overrides a DetectorParams or TrackerParams field, e.g.
//   tracker=kf min_hits=3 cluster_tolerance_m=0.7      (tracker = none | nn | kf)
// Outputs:
//   <out_dir>/detections.csv  stamp_ns,track_id,cx,cy,cz,length,width,height,yaw,num_points,hits  (ego frame)
//   <out_dir>/timings.csv     stamp_ns,n_in,n_nonground,n_voxel,n_clusters,n_boxes,n_out,ground_ms,voxel_ms,cluster_ms,fit_ms,detect_ms,track_ms
//   <out_dir>/params.txt      the parameters actually used
//
// The tracker runs in the map frame using the /tf (map -> base_link) message that
// precedes each cloud in the bag with the same stamp.
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <vector>
#include <map>
#include <string>

#include "lidar_tracking/detector.hpp"
#include "lidar_tracking/monitor.hpp"
#include "lidar_tracking/ros_conversions.hpp"
#include "lidar_tracking/tracker.hpp"

using lidar_tracking::DetectorParams;
using lidar_tracking::TrackerParams;

namespace
{

struct AllParams
{
  DetectorParams det;
  TrackerParams trk;
  std::string ground_mask_dir;  // map-ground oracle masks (scripts/oracle_ground_masks.py)
};

using Setter = std::function<void (AllParams &, const std::string &)>;

Setter num(std::function<void(AllParams &, double)> f)
{
  return [f](AllParams & p, const std::string & v) {f(p, std::stod(v));};
}

// name -> setter, so the CLI and the structs cannot drift apart silently.
std::map<std::string, Setter> param_setters()
{
  return {
    {"max_range_m", num([](AllParams & p, double v) {p.det.max_range_m = v;})},
    {"ground_cell_m", num([](AllParams & p, double v) {p.det.ground_cell_m = v;})},
    {"ground_neighbourhood", num([](AllParams & p, double v) {p.det.ground_neighbourhood = static_cast<int>(v);})},
    {"ground_clearance_m", num([](AllParams & p, double v) {p.det.ground_clearance_m = v;})},
    {"max_height_m", num([](AllParams & p, double v) {p.det.max_height_m = v;})},
    {"voxel_m", num([](AllParams & p, double v) {p.det.voxel_m = v;})},
    {"cluster_2d", num([](AllParams & p, double v) {p.det.cluster_2d = v != 0.0;})},
    {"cluster_method", num([](AllParams & p, double v) {p.det.cluster_method = static_cast<int>(v);})},
    {"cluster_tolerance_m", num([](AllParams & p, double v) {p.det.cluster_tolerance_m = v;})},
    {"min_cluster_points", num([](AllParams & p, double v) {p.det.min_cluster_points = static_cast<int>(v);})},
    {"max_cluster_points", num([](AllParams & p, double v) {p.det.max_cluster_points = static_cast<int>(v);})},
    {"max_box_length_m", num([](AllParams & p, double v) {p.det.max_box_length_m = v;})},
    {"max_box_area_m2", num([](AllParams & p, double v) {p.det.max_box_area_m2 = v;})},
    {"max_clearance_m", num([](AllParams & p, double v) {p.det.max_clearance_m = v;})},
    {"min_box_width_m", num([](AllParams & p, double v) {p.det.min_box_width_m = v;})},
    {"min_box_height_m", num([](AllParams & p, double v) {p.det.min_box_height_m = v;})},
    {"ground_mask_dir", [](AllParams & p, const std::string & v) {p.ground_mask_dir = v; p.det.ground_method = 1;}},
    {"tracker", [](AllParams & p, const std::string & v) {
        if (v == "none") {p.trk.mode = TrackerParams::Mode::None;} else if (v == "nn") {
          p.trk.mode = TrackerParams::Mode::Nearest;
        } else if (v == "kf") {p.trk.mode = TrackerParams::Mode::Kalman;} else {
          throw std::invalid_argument("tracker must be none, nn or kf");
        }
      }},
    {"gate_m", num([](AllParams & p, double v) {p.trk.gate_m = v;})},
    {"min_hits", num([](AllParams & p, double v) {p.trk.min_hits = static_cast<int>(v);})},
    {"max_misses", num([](AllParams & p, double v) {p.trk.max_misses = static_cast<int>(v);})},
    {"accel_std", num([](AllParams & p, double v) {p.trk.accel_std = v;})},
    {"meas_std_m", num([](AllParams & p, double v) {p.trk.meas_std_m = v;})},
    {"output_filtered", num([](AllParams & p, double v) {p.trk.output_filtered = v != 0.0;})},
  };
}

const char * mode_name(TrackerParams::Mode m)
{
  switch (m) {
    case TrackerParams::Mode::None: return "none";
    case TrackerParams::Mode::Nearest: return "nn";
    default: return "kf";
  }
}

void write_params(std::ostream & os, const AllParams & a)
{
  const auto & p = a.det;
  const auto & t = a.trk;
  os << "max_range_m=" << p.max_range_m << "\nground_cell_m=" << p.ground_cell_m
     << "\nground_neighbourhood=" << p.ground_neighbourhood << "\nground_clearance_m=" << p.ground_clearance_m
     << "\nmax_height_m=" << p.max_height_m << "\nvoxel_m=" << p.voxel_m << "\ncluster_2d=" << p.cluster_2d
     << "\ncluster_method=" << p.cluster_method << "\ncluster_tolerance_m=" << p.cluster_tolerance_m << "\nmin_cluster_points=" << p.min_cluster_points
     << "\nmax_cluster_points=" << p.max_cluster_points << "\nmax_box_length_m=" << p.max_box_length_m
     << "\nmax_box_area_m2=" << p.max_box_area_m2 << "\nmin_box_height_m=" << p.min_box_height_m
     << "\nmax_clearance_m=" << p.max_clearance_m << "\nmin_box_width_m=" << p.min_box_width_m
     << "\ntracker=" << mode_name(t.mode) << "\ngate_m=" << t.gate_m << "\nmin_hits=" << t.min_hits
     << "\nmax_misses=" << t.max_misses << "\naccel_std=" << t.accel_std << "\nmeas_std_m=" << t.meas_std_m
     << "\noutput_filtered=" << t.output_filtered << "\nground_method=" << p.ground_method
     << "\nground_mask_dir=" << a.ground_mask_dir << "\n";
}

int64_t to_ns(const builtin_interfaces::msg::Time & t)
{
  return static_cast<int64_t>(t.sec) * 1000000000LL + t.nanosec;
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::cerr << "usage: process_bag <bag_dir> <out_dir> [key=value ...]\n";
    return 2;
  }
  const std::string bag_dir = argv[1];
  const std::filesystem::path out_dir = argv[2];

  AllParams params;
  params.trk.mode = TrackerParams::Mode::None;  // default: the no-tracking baseline
  const auto setters = param_setters();
  for (int i = 3; i < argc; ++i) {
    const std::string kv = argv[i];
    const auto eq = kv.find('=');
    const auto it = eq == std::string::npos ? setters.end() : setters.find(kv.substr(0, eq));
    if (it == setters.end()) {
      std::cerr << "unknown parameter: " << kv << "\n";
      return 2;
    }
    it->second(params, kv.substr(eq + 1));
  }

  std::filesystem::create_directories(out_dir);
  {
    std::ofstream pf(out_dir / "params.txt");
    write_params(pf, params);
  }
  std::ofstream det(out_dir / "detections.csv");
  std::ofstream tim(out_dir / "timings.csv");
  det << "stamp_ns,track_id,cx,cy,cz,length,width,height,yaw,num_points,hits,clearance,flags\n";
  tim << "stamp_ns,n_in,n_nonground,n_voxel,n_clusters,n_boxes,n_out,ground_ms,voxel_ms,cluster_ms,fit_ms,detect_ms,track_ms\n";

  lidar_tracking::Detector detector(params.det);
  lidar_tracking::Tracker tracker(params.trk);
  lidar_tracking::TrackMonitor monitor;  // a-priori thresholds, see monitor.hpp
  rosbag2_cpp::Reader reader;
  reader.open(bag_dir);
  rclcpp::Serialization<sensor_msgs::msg::PointCloud2> cloud_ser;
  rclcpp::Serialization<tf2_msgs::msg::TFMessage> tf_ser;
  lidar_tracking::Pose2D ego;
  int64_t ego_stamp = -1;
  int frames = 0;
  int64_t n_out_total = 0;
  double total_ms = 0.0;
  char buf[256];

  while (reader.has_next()) {
    auto bag_msg = reader.read_next();
    rclcpp::SerializedMessage serialized(*bag_msg->serialized_data);
    if (bag_msg->topic_name == "/tf") {
      tf2_msgs::msg::TFMessage tf;
      tf_ser.deserialize_message(&serialized, &tf);
      for (const auto & t : tf.transforms) {
        if (t.header.frame_id != "map" || t.child_frame_id != "base_link") {continue;}
        const auto & q = t.transform.rotation;
        ego.x = t.transform.translation.x;
        ego.y = t.transform.translation.y;
        ego.yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        ego_stamp = to_ns(t.header.stamp);
      }
      continue;
    }
    if (bag_msg->topic_name != "/av2/lidar") {continue;}
    sensor_msgs::msg::PointCloud2 cloud;
    cloud_ser.deserialize_message(&serialized, &cloud);
    const int64_t stamp = to_ns(cloud.header.stamp);
    if (ego_stamp != stamp) {
      std::cerr << "no /tf with the cloud's stamp " << stamp << " (last tf " << ego_stamp << ")\n";
      return 1;
    }

    auto points = lidar_tracking::from_cloud2(cloud);
    if (!params.ground_mask_dir.empty()) {
      const auto log_id = std::filesystem::path(bag_dir).lexically_normal().filename().string().empty() ?
        std::filesystem::path(bag_dir).lexically_normal().parent_path().filename().string() :
        std::filesystem::path(bag_dir).lexically_normal().filename().string();
      const auto mask_path = std::filesystem::path(params.ground_mask_dir) / log_id / (std::to_string(stamp) + ".bin");
      std::ifstream mf(mask_path, std::ios::binary);
      std::vector<char> mask((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
      if (mask.size() != points.size()) {
        std::cerr << "mask " << mask_path << " has " << mask.size() << " entries, cloud has " << points.size() << "\n";
        return 1;
      }
      std::vector<lidar_tracking::Point> kept;
      for (size_t i = 0; i < points.size(); ++i) {
        if (mask[i]) {kept.push_back(points[i]);}
      }
      points.swap(kept);
    }
    lidar_tracking::StageTimings t;
    const auto boxes = detector.detect(points, &t);
    const auto t0 = std::chrono::steady_clock::now();
    const auto tracks = tracker.update(boxes, ego, static_cast<double>(stamp) * 1e-9);
    const double track_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const auto flags = monitor.check(tracks, ego, static_cast<double>(stamp) * 1e-9);
    for (size_t k = 0; k < tracks.size(); ++k) {
      const auto & o = tracks[k];
      const auto & b = o.box;
      std::snprintf(
        buf, sizeof(buf), "%lld,%lld,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%d,%d,%.3f,%d\n",
        static_cast<long long>(stamp), static_cast<long long>(o.id),
        b.cx, b.cy, b.cz, b.length, b.width, b.height, b.yaw, b.num_points, o.hits, b.clearance, flags[k]);
      det << buf;
    }
    std::snprintf(
      buf, sizeof(buf), "%lld,%d,%d,%d,%d,%d,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
      static_cast<long long>(stamp), t.n_in, t.n_nonground, t.n_voxel, t.n_clusters, t.n_boxes, tracks.size(),
      t.ground_ms, t.voxel_ms, t.cluster_ms, t.fit_ms, t.total_ms, track_ms);
    tim << buf;
    ++frames;
    n_out_total += static_cast<int64_t>(tracks.size());
    total_ms += t.total_ms + track_ms;
  }
  const auto & mc = monitor.counts();
  std::cout << "frames=" << frames << " outputs=" << n_out_total << " monitor_flags=" << mc.any
            << " mean_ms=" << (frames ? total_ms / frames : 0.0) << "\n";
  return frames > 0 ? 0 : 1;
}
