// Minimum-cost assignment (Hungarian / Kuhn-Munkres, O(n^2 m)) for rectangular matrices.
#pragma once

#include <vector>

namespace lidar_tracking
{

// cost is rows x cols, row-major. Returns, for each row, the assigned column or -1.
// Every row gets a column when rows <= cols; otherwise rows - cols rows get -1.
// Gating is the caller's job: give forbidden pairs a large cost and drop them after.
std::vector<int> hungarian(const std::vector<double> & cost, int rows, int cols);

}  // namespace lidar_tracking
