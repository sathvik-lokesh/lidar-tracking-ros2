// Bird's-eye-view box fitting for a cluster of points. Pure C++, no PCL.
#pragma once

#include <utility>
#include <vector>

#include "lidar_tracking/types.hpp"

namespace lidar_tracking
{

using Point2 = std::pair<double, double>;

// Andrew's monotone chain. Returns the hull counter-clockwise without repeating
// the first point. Collinear points are dropped. Fewer than 3 unique points are
// returned as-is (deduplicated).
std::vector<Point2> convex_hull(std::vector<Point2> pts);

// Minimum-area enclosing rectangle via rotating calipers over the hull edges.
// The returned box has length >= width and yaw in (-pi/2, pi/2]; z fields are
// filled from the points' z range. num_points = pts.size().
Box fit_min_area_box(const std::vector<Point> & pts);

}  // namespace lidar_tracking
