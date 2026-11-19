// Offline runner: reads every cloud from a bag in order (no dropped frames, no
// wall-clock dependence), runs the detector, and writes CSVs for scoring.
//
//   process_bag <bag_dir> <out_dir> [key=value ...]
//
// key=value overrides any DetectorParams field, e.g. cluster_tolerance_m=0.7.
// Outputs:
//   <out_dir>/detections.csv  stamp_ns,track_id,cx,cy,cz,length,width,height,yaw,num_points  (ego frame)
//   <out_dir>/timings.csv     stamp_ns,n_in,n_nonground,n_voxel,n_clusters,n_boxes,ground_ms,voxel_ms,cluster_ms,fit_ms,total_ms
//   <out_dir>/params.txt      the parameters actually used
//
// Tracking is not done here yet: every detection gets a fresh track id, which is
// the "no tracking" baseline.
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>

#include "lidar_tracking/detector.hpp"
#include "lidar_tracking/ros_conversions.hpp"

using lidar_tracking::DetectorParams;

namespace
{

// name -> setter, so the CLI, params.txt and the struct cannot drift apart.
std::map<std::string, std::function<void(DetectorParams &, double)>> param_setters()
{
  return {
    {"max_range_m", [](DetectorParams & p, double v) {p.max_range_m = v;}},
    {"ground_cell_m", [](DetectorParams & p, double v) {p.ground_cell_m = v;}},
    {"ground_neighbourhood", [](DetectorParams & p, double v) {p.ground_neighbourhood = static_cast<int>(v);}},
    {"ground_clearance_m", [](DetectorParams & p, double v) {p.ground_clearance_m = v;}},
    {"max_height_m", [](DetectorParams & p, double v) {p.max_height_m = v;}},
    {"voxel_m", [](DetectorParams & p, double v) {p.voxel_m = v;}},
    {"cluster_2d", [](DetectorParams & p, double v) {p.cluster_2d = v != 0.0;}},
    {"cluster_tolerance_m", [](DetectorParams & p, double v) {p.cluster_tolerance_m = v;}},
    {"min_cluster_points", [](DetectorParams & p, double v) {p.min_cluster_points = static_cast<int>(v);}},
    {"max_cluster_points", [](DetectorParams & p, double v) {p.max_cluster_points = static_cast<int>(v);}},
    {"max_box_length_m", [](DetectorParams & p, double v) {p.max_box_length_m = v;}},
    {"max_box_area_m2", [](DetectorParams & p, double v) {p.max_box_area_m2 = v;}},
    {"min_box_height_m", [](DetectorParams & p, double v) {p.min_box_height_m = v;}},
  };
}

void write_params(std::ostream & os, const DetectorParams & p)
{
  os << "max_range_m=" << p.max_range_m << "\nground_cell_m=" << p.ground_cell_m
     << "\nground_neighbourhood=" << p.ground_neighbourhood << "\nground_clearance_m=" << p.ground_clearance_m
     << "\nmax_height_m=" << p.max_height_m << "\nvoxel_m=" << p.voxel_m << "\ncluster_2d=" << p.cluster_2d
     << "\ncluster_tolerance_m=" << p.cluster_tolerance_m << "\nmin_cluster_points=" << p.min_cluster_points
     << "\nmax_cluster_points=" << p.max_cluster_points << "\nmax_box_length_m=" << p.max_box_length_m
     << "\nmax_box_area_m2=" << p.max_box_area_m2 << "\nmin_box_height_m=" << p.min_box_height_m << "\n";
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

  DetectorParams params;
  const auto setters = param_setters();
  for (int i = 3; i < argc; ++i) {
    const std::string kv = argv[i];
    const auto eq = kv.find('=');
    const auto it = eq == std::string::npos ? setters.end() : setters.find(kv.substr(0, eq));
    if (it == setters.end()) {
      std::cerr << "unknown parameter: " << kv << "\n";
      return 2;
    }
    it->second(params, std::stod(kv.substr(eq + 1)));
  }

  std::filesystem::create_directories(out_dir);
  {
    std::ofstream pf(out_dir / "params.txt");
    write_params(pf, params);
  }
  std::ofstream det(out_dir / "detections.csv");
  std::ofstream tim(out_dir / "timings.csv");
  det << "stamp_ns,track_id,cx,cy,cz,length,width,height,yaw,num_points\n";
  tim << "stamp_ns,n_in,n_nonground,n_voxel,n_clusters,n_boxes,ground_ms,voxel_ms,cluster_ms,fit_ms,total_ms\n";

  lidar_tracking::Detector detector(params);
  rosbag2_cpp::Reader reader;
  reader.open(bag_dir);
  rclcpp::Serialization<sensor_msgs::msg::PointCloud2> ser;
  int64_t next_id = 0;
  int frames = 0;
  double total_ms = 0.0;
  char buf[256];

  while (reader.has_next()) {
    auto bag_msg = reader.read_next();
    if (bag_msg->topic_name != "/av2/lidar") {continue;}
    rclcpp::SerializedMessage serialized(*bag_msg->serialized_data);
    sensor_msgs::msg::PointCloud2 cloud;
    ser.deserialize_message(&serialized, &cloud);
    const int64_t stamp = static_cast<int64_t>(cloud.header.stamp.sec) * 1000000000LL + cloud.header.stamp.nanosec;

    lidar_tracking::StageTimings t;
    const auto boxes = detector.detect(lidar_tracking::from_cloud2(cloud), &t);
    for (const auto & b : boxes) {
      std::snprintf(
        buf, sizeof(buf), "%lld,%lld,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%d\n",
        static_cast<long long>(stamp), static_cast<long long>(next_id++),
        b.cx, b.cy, b.cz, b.length, b.width, b.height, b.yaw, b.num_points);
      det << buf;
    }
    std::snprintf(
      buf, sizeof(buf), "%lld,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n",
      static_cast<long long>(stamp), t.n_in, t.n_nonground, t.n_voxel, t.n_clusters, t.n_boxes,
      t.ground_ms, t.voxel_ms, t.cluster_ms, t.fit_ms, t.total_ms);
    tim << buf;
    ++frames;
    total_ms += t.total_ms;
  }
  std::cout << "frames=" << frames << " detections=" << next_id
            << " mean_total_ms=" << (frames ? total_ms / frames : 0.0) << "\n";
  return frames > 0 ? 0 : 1;
}
