#include "lidar_tracking/grid_cluster.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace lidar_tracking
{

namespace
{
struct UnionFind
{
  std::vector<int> parent;
  explicit UnionFind(int n)
  : parent(n) {std::iota(parent.begin(), parent.end(), 0);}
  int find(int i)
  {
    while (parent[i] != i) {parent[i] = parent[parent[i]]; i = parent[i];}
    return i;
  }
  void unite(int a, int b)
  {
    a = find(a); b = find(b);
    if (a != b) {parent[std::max(a, b)] = std::min(a, b);}
  }
};

int64_t key(int ix, int iy)
{
  return (static_cast<int64_t>(ix) << 32) ^ static_cast<uint32_t>(iy);
}
}  // namespace

std::vector<std::vector<GridCell>> grid_clusters(
  const std::vector<Point> & pts, double cell_m, double tolerance_m, int min_cells, int max_cells)
{
  // 1. Bin points into cells.
  std::unordered_map<int64_t, int> index;
  index.reserve(pts.size());
  std::vector<std::pair<int, int>> ij;
  std::vector<GridCell> cells;
  for (const auto & p : pts) {
    const int ix = static_cast<int>(std::floor(p.x / cell_m));
    const int iy = static_cast<int>(std::floor(p.y / cell_m));
    auto [it, inserted] = index.try_emplace(key(ix, iy), static_cast<int>(cells.size()));
    if (inserted) {
      cells.push_back({(ix + 0.5) * cell_m, (iy + 0.5) * cell_m, p.z, p.z, 0});
      ij.emplace_back(ix, iy);
    }
    GridCell & c = cells[it->second];
    c.zmin = std::min(c.zmin, p.z);
    c.zmax = std::max(c.zmax, p.z);
    c.count += 1;
  }

  // 2. Neighbour offsets whose squares come within the tolerance. Squares offset by
  // (dx, dy) cells are (|dx|-1)*cell apart along x (0 if adjacent or same column).
  std::vector<std::pair<int, int>> offsets;
  const int r = static_cast<int>(std::ceil(tolerance_m / cell_m)) + 1;
  for (int dx = 0; dx <= r; ++dx) {
    for (int dy = -r; dy <= r; ++dy) {
      if (dx == 0 && dy <= 0) {continue;}  // each unordered pair once
      const double gx = std::max(0, std::abs(dx) - 1) * cell_m;
      const double gy = std::max(0, std::abs(dy) - 1) * cell_m;
      if (std::hypot(gx, gy) <= tolerance_m) {offsets.emplace_back(dx, dy);}
    }
  }

  // 3. Union occupied neighbours.
  UnionFind uf(static_cast<int>(cells.size()));
  for (size_t c = 0; c < cells.size(); ++c) {
    const auto [ix, iy] = ij[c];
    for (const auto & [dx, dy] : offsets) {
      const auto it = index.find(key(ix + dx, iy + dy));
      if (it != index.end()) {uf.unite(static_cast<int>(c), it->second);}
    }
  }

  // 4. Collect.
  std::unordered_map<int, int> root_to_cluster;
  std::vector<std::vector<GridCell>> clusters;
  for (size_t c = 0; c < cells.size(); ++c) {
    const int root = uf.find(static_cast<int>(c));
    auto [it, inserted] = root_to_cluster.try_emplace(root, static_cast<int>(clusters.size()));
    if (inserted) {clusters.emplace_back();}
    clusters[it->second].push_back(cells[c]);
  }
  std::vector<std::vector<GridCell>> out;
  for (auto & cl : clusters) {
    const int n = static_cast<int>(cl.size());
    if (n >= min_cells && n <= max_cells) {out.push_back(std::move(cl));}
  }
  return out;
}

}  // namespace lidar_tracking
