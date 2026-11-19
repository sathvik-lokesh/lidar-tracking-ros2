#include "lidar_tracking/box_fit.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lidar_tracking
{

namespace
{
double cross(const Point2 & o, const Point2 & a, const Point2 & b)
{
  return (a.first - o.first) * (b.second - o.second) - (a.second - o.second) * (b.first - o.first);
}
}  // namespace

std::vector<Point2> convex_hull(std::vector<Point2> pts)
{
  std::sort(pts.begin(), pts.end());
  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
  if (pts.size() < 3) {
    return pts;
  }
  std::vector<Point2> hull(2 * pts.size());
  size_t k = 0;
  for (const auto & p : pts) {  // lower hull
    while (k >= 2 && cross(hull[k - 2], hull[k - 1], p) <= 0) {--k;}
    hull[k++] = p;
  }
  for (size_t i = pts.size() - 1, t = k + 1; i-- > 0; ) {  // upper hull
    while (k >= t && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) {--k;}
    hull[k++] = pts[i];
  }
  hull.resize(k - 1);
  return hull;
}

Box fit_min_area_box(const std::vector<Point> & pts)
{
  Box box;
  box.num_points = static_cast<int>(pts.size());
  if (pts.empty()) {
    return box;
  }
  double zmin = std::numeric_limits<double>::max();
  double zmax = std::numeric_limits<double>::lowest();
  std::vector<Point2> xy;
  xy.reserve(pts.size());
  for (const auto & p : pts) {
    xy.emplace_back(p.x, p.y);
    zmin = std::min(zmin, static_cast<double>(p.z));
    zmax = std::max(zmax, static_cast<double>(p.z));
  }
  box.cz = 0.5 * (zmin + zmax);
  box.height = zmax - zmin;

  const auto hull = convex_hull(std::move(xy));
  if (hull.size() == 1) {
    box.cx = hull[0].first;
    box.cy = hull[0].second;
    return box;
  }

  // Try every hull edge direction; the minimum-area rectangle has one side
  // collinear with a hull edge. For 2 points the single edge is used.
  double best_area = std::numeric_limits<double>::max();
  const size_t n = hull.size();
  const size_t n_edges = (n == 2) ? 1 : n;
  for (size_t i = 0; i < n_edges; ++i) {
    const auto & a = hull[i];
    const auto & b = hull[(i + 1) % n];
    const double theta = std::atan2(b.second - a.second, b.first - a.first);
    const double c = std::cos(theta), s = std::sin(theta);
    double umin = std::numeric_limits<double>::max(), umax = std::numeric_limits<double>::lowest();
    double vmin = umin, vmax = umax;
    for (const auto & p : hull) {
      const double u = c * p.first + s * p.second;
      const double v = -s * p.first + c * p.second;
      umin = std::min(umin, u); umax = std::max(umax, u);
      vmin = std::min(vmin, v); vmax = std::max(vmax, v);
    }
    const double du = umax - umin, dv = vmax - vmin;
    // Tie-break on the perimeter so degenerate (zero-area) clusters still get the
    // tightest box along their extent.
    const double area = du * dv + 1e-9 * (du + dv);
    if (area < best_area) {
      best_area = area;
      const double uc = 0.5 * (umin + umax), vc = 0.5 * (vmin + vmax);
      box.cx = c * uc - s * vc;
      box.cy = s * uc + c * vc;
      if (du >= dv) {
        box.length = du; box.width = dv; box.yaw = theta;
      } else {
        box.length = dv; box.width = du; box.yaw = theta + M_PI / 2.0;
      }
    }
  }
  // Normalise yaw to (-pi/2, pi/2]: a box is symmetric under a half turn.
  while (box.yaw > M_PI / 2.0) {box.yaw -= M_PI;}
  while (box.yaw <= -M_PI / 2.0) {box.yaw += M_PI;}
  return box;
}

}  // namespace lidar_tracking
