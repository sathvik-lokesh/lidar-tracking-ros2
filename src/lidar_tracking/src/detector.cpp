#include "lidar_tracking/detector.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "lidar_tracking/box_fit.hpp"
#include "lidar_tracking/grid_cluster.hpp"

namespace lidar_tracking
{

namespace
{
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
constexpr float kInf = std::numeric_limits<float>::infinity();
}  // namespace

struct Detector::Impl
{
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree{new pcl::search::KdTree<pcl::PointXYZ>};
};

Detector::Detector(DetectorParams params)
: params_(params), impl_(std::make_unique<Impl>()) {}

Detector::~Detector() = default;

std::vector<Point> Detector::remove_ground(const std::vector<Point> & cloud) const
{
  const double R = params_.max_range_m;
  const double cell = params_.ground_cell_m;
  const int n = static_cast<int>(std::ceil(2.0 * R / cell));
  auto idx = [&](double v) {return std::clamp(static_cast<int>((v + R) / cell), 0, n - 1);};

  // 1. Per-cell minimum z of points inside the range.
  std::vector<float> cell_min(static_cast<size_t>(n) * n, kInf);
  std::vector<int> point_cell(cloud.size(), -1);
  const double r2 = R * R;
  for (size_t i = 0; i < cloud.size(); ++i) {
    const auto & p = cloud[i];
    if (static_cast<double>(p.x) * p.x + static_cast<double>(p.y) * p.y > r2) {continue;}
    const int c = idx(p.x) * n + idx(p.y);
    point_cell[i] = c;
    cell_min[c] = std::min(cell_min[c], p.z);
  }

  // 2. Ground height = min over a (2k+1)^2 neighbourhood. Separable: rows then columns.
  const int k = params_.ground_neighbourhood;
  std::vector<float> tmp(cell_min.size(), kInf), ground(cell_min.size(), kInf);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float m = kInf;
      for (int d = std::max(0, j - k); d <= std::min(n - 1, j + k); ++d) {m = std::min(m, cell_min[i * n + d]);}
      tmp[i * n + j] = m;
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float m = kInf;
      for (int d = std::max(0, i - k); d <= std::min(n - 1, i + k); ++d) {m = std::min(m, tmp[d * n + j]);}
      ground[i * n + j] = m;
    }
  }

  ground_ = ground;
  ground_n_ = n;

  // 3. Keep points in the band (ground + clearance, ground + max_height].
  std::vector<Point> out;
  out.reserve(cloud.size() / 2);
  for (size_t i = 0; i < cloud.size(); ++i) {
    if (point_cell[i] < 0) {continue;}
    const float h = cloud[i].z - ground[point_cell[i]];
    if (h > params_.ground_clearance_m && h <= params_.max_height_m) {out.push_back(cloud[i]);}
  }
  return out;
}

std::vector<Box> Detector::detect(const std::vector<Point> & cloud, StageTimings * timings)
{
  StageTimings t;
  const auto t_start = Clock::now();
  t.n_in = static_cast<int>(cloud.size());

  auto t0 = Clock::now();
  std::vector<Point> above;
  if (params_.ground_method == 1) {
    const double r2 = params_.max_range_m * params_.max_range_m;
    for (const auto & p : cloud) {
      if (static_cast<double>(p.x) * p.x + static_cast<double>(p.y) * p.y <= r2) {above.push_back(p);}
    }
    ground_n_ = 0;  // no ground grid: clearance is reported as 0
  } else {
    above = remove_ground(cloud);
  }
  t.ground_ms = ms_since(t0);
  t.n_nonground = static_cast<int>(above.size());

  if (params_.cluster_method == 1) {
    detect_grid(above, t);
  } else {
    detect_pcl(above, t);
  }
  t.total_ms = ms_since(t_start);
  if (timings) {*timings = t;}
  return std::move(t_boxes_);
}

void Detector::detect_grid(const std::vector<Point> & above, StageTimings & t)
{
  t_boxes_.clear();
  auto t0 = Clock::now();
  const auto clusters = grid_clusters(
    above, params_.voxel_m, params_.cluster_tolerance_m, params_.min_cluster_points, params_.max_cluster_points);
  t.cluster_ms = ms_since(t0);
  t.n_clusters = static_cast<int>(clusters.size());
  t0 = Clock::now();
  std::vector<Point> members;
  for (const auto & cl : clusters) {
    members.clear();
    for (const auto & c : cl) {
      members.push_back({static_cast<float>(c.cx), static_cast<float>(c.cy), c.zmin, 0.f});
      members.push_back({static_cast<float>(c.cx), static_cast<float>(c.cy), c.zmax, 0.f});
    }
    Box b = fit_min_area_box(members);
    b.num_points = static_cast<int>(cl.size());
    if (keep_box(b)) {t_boxes_.push_back(b);}
  }
  t.fit_ms = ms_since(t0);
  t.n_boxes = static_cast<int>(t_boxes_.size());
}

double Detector::ground_at(double x, double y) const
{
  const double R = params_.max_range_m, cell = params_.ground_cell_m;
  if (ground_n_ == 0) {return std::numeric_limits<double>::quiet_NaN();}
  const int i = std::clamp(static_cast<int>((x + R) / cell), 0, ground_n_ - 1);
  const int j = std::clamp(static_cast<int>((y + R) / cell), 0, ground_n_ - 1);
  const float g = ground_[static_cast<size_t>(i) * ground_n_ + j];
  return std::isfinite(g) ? g : std::numeric_limits<double>::quiet_NaN();
}

bool Detector::keep_box(Box & b) const
{
  const double g = ground_at(b.cx, b.cy);
  b.clearance = std::isfinite(g) ? (b.cz - 0.5 * b.height) - g : 0.0;
  return b.length <= params_.max_box_length_m && b.length * b.width <= params_.max_box_area_m2 &&
         b.height >= params_.min_box_height_m && b.width >= params_.min_box_width_m &&
         b.clearance <= params_.max_clearance_m;
}

void Detector::detect_pcl(const std::vector<Point> & above, StageTimings & t)
{
  t_boxes_.clear();
  // Voxel grid in 3D. The centroid of each voxel stands in for its points.
  auto t0 = Clock::now();
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pc->reserve(above.size());
  for (const auto & p : above) {pc->emplace_back(p.x, p.y, p.z);}
  if (params_.voxel_m > 0.0 && !pc->empty()) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(pc);
    const float v = static_cast<float>(params_.voxel_m);
    vg.setLeafSize(v, v, v);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds(new pcl::PointCloud<pcl::PointXYZ>);
    vg.filter(*ds);
    pc = ds;
  }
  t.voxel_ms = ms_since(t0);
  t.n_voxel = static_cast<int>(pc->size());

  // Euclidean clustering; in 2D mode the search runs on a flattened copy.
  t0 = Clock::now();
  std::vector<pcl::PointIndices> clusters;
  if (!pc->empty()) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr search_pc = pc;
    if (params_.cluster_2d) {
      search_pc.reset(new pcl::PointCloud<pcl::PointXYZ>(*pc));
      for (auto & p : search_pc->points) {p.z = 0.f;}
    }
    impl_->tree->setInputCloud(search_pc);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(params_.cluster_tolerance_m);
    ec.setMinClusterSize(params_.min_cluster_points);
    ec.setMaxClusterSize(params_.max_cluster_points);
    ec.setSearchMethod(impl_->tree);
    ec.setInputCloud(search_pc);
    ec.extract(clusters);
  }
  t.cluster_ms = ms_since(t0);
  t.n_clusters = static_cast<int>(clusters.size());

  // Fit a box per cluster (on the 3D voxel centroids) and apply sanity limits.
  t0 = Clock::now();
  std::vector<Box> boxes;
  boxes.reserve(clusters.size());
  std::vector<Point> members;
  for (const auto & c : clusters) {
    members.clear();
    for (int i : c.indices) {
      const auto & p = pc->points[i];
      members.push_back({p.x, p.y, p.z, 0.f});
    }
    Box b = fit_min_area_box(members);
    if (keep_box(b)) {boxes.push_back(b);}
  }
  t.fit_ms = ms_since(t0);
  t.n_boxes = static_cast<int>(boxes.size());
  t_boxes_ = std::move(boxes);
}

}  // namespace lidar_tracking
