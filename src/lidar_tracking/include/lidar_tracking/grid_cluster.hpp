// BEV grid connected-components clustering. Pure C++, no PCL.
//
// Points are binned into square BEV cells. Two occupied cells are linked when the
// closest distance between their squares is <= tolerance, so any two points closer
// than the tolerance always end up in the same cluster (no false splits). Points up
// to tolerance + 2 * cell * sqrt(2) apart may be merged, so keep cells small.
#pragma once

#include <vector>

#include "lidar_tracking/types.hpp"

namespace lidar_tracking
{

struct GridCell
{
  double cx{0.0};  // cell centre
  double cy{0.0};
  float zmin{0.f};
  float zmax{0.f};
  int count{0};
};

// Returns clusters as lists of cells; clusters with fewer than min_cells or more
// than max_cells occupied cells are dropped.
std::vector<std::vector<GridCell>> grid_clusters(
  const std::vector<Point> & pts, double cell_m, double tolerance_m, int min_cells, int max_cells);

}  // namespace lidar_tracking
