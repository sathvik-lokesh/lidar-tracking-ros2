// Plain data types shared by the detector, tracker, ROS node and offline tools.
// No ROS or PCL here, so the core library can be unit-tested and reused anywhere.
#pragma once

#include <cstdint>
#include <vector>

namespace lidar_tracking
{

struct Point
{
  float x{0.f};
  float y{0.f};
  float z{0.f};
  float intensity{0.f};
};

// Oriented 3D box in the sensor/ego frame. Yaw is about +z; length runs along yaw.
struct Box
{
  double cx{0.0};
  double cy{0.0};
  double cz{0.0};
  double length{0.0};
  double width{0.0};
  double height{0.0};
  double yaw{0.0};
  int num_points{0};
  // Height of the box bottom above the local ground estimate (detector output; 0 if unknown).
  double clearance{0.0};
};

struct DetectorParams
{
  // Region of interest. AV2 tracking is scored to 50 m from the ego; keep a margin
  // so objects whose centre is just inside 50 m are not cut in half.
  double max_range_m{55.0};

  // Ground removal: per-cell minimum z on a BEV grid, then the ground height of a
  // cell is the minimum over a (2k+1)x(2k+1) neighbourhood, so cells fully covered
  // by a vehicle still see the road beside it.
  double ground_cell_m{1.0};
  int ground_neighbourhood{2};  // k: 2 -> 5x5 cells
  double ground_clearance_m{0.25};  // points lower than ground + this are ground
  double max_height_m{3.0};  // drop points higher than ground + this (canopy, overhangs)

  double voxel_m{0.15};  // 0 disables down-sampling

  // Euclidean clustering. In 2D mode z is ignored, so vertical gaps between lidar
  // beams on distant objects do not split them.
  bool cluster_2d{true};
  // 0 = PCL 3D voxel grid + KD-tree Euclidean clustering (2D if cluster_2d).
  // 1 = BEV grid connected components (grid_cluster.hpp): cell size = voxel_m,
  //     min/max_cluster_points count occupied cells. Always 2D.
  int cluster_method{0};
  double cluster_tolerance_m{0.5};
  int min_cluster_points{5};
  int max_cluster_points{30000};

  // Box sanity limits (articulated buses are ~18 m).
  double max_box_length_m{20.0};
  double max_box_area_m2{60.0};
  double min_box_height_m{0.0};
  double max_clearance_m{1e9};  // drop boxes floating higher than this above ground (off by default)
  double min_box_width_m{0.0};  // drop boxes thinner than this (off by default)
};

struct StageTimings
{
  double ground_ms{0.0};
  double voxel_ms{0.0};
  double cluster_ms{0.0};
  double fit_ms{0.0};
  double total_ms{0.0};
  int n_in{0};
  int n_nonground{0};
  int n_voxel{0};
  int n_clusters{0};
  int n_boxes{0};
};

}  // namespace lidar_tracking
