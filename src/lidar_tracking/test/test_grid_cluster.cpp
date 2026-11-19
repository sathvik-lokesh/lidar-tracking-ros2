#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "lidar_tracking/grid_cluster.hpp"

using lidar_tracking::Point;
using lidar_tracking::grid_clusters;

TEST(GridCluster, NeverSplitsPointsWithinTolerance)
{
  // Random pairs at distance < tolerance must always share a cluster.
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> pos(-20.0, 20.0), ang(0.0, 6.283), frac(0.0, 0.999);
  for (int i = 0; i < 2000; ++i) {
    const double x = pos(rng), y = pos(rng), a = ang(rng), d = 0.5 * frac(rng);
    const std::vector<Point> pts{{float(x), float(y), 0.f, 0.f}, {float(x + d * std::cos(a)), float(y + d * std::sin(a)), 1.f, 0.f}};
    const auto cl = grid_clusters(pts, 0.15, 0.5, 1, 1000);
    ASSERT_EQ(cl.size(), 1u) << "split pair at distance " << d;
  }
}

TEST(GridCluster, SeparatesDistantPoints)
{
  // Beyond tolerance + 2 * cell * sqrt(2) a pair can never merge.
  const double far = 0.5 + 2 * 0.15 * std::sqrt(2.0) + 0.01;
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> pos(-20.0, 20.0), ang(0.0, 6.283);
  for (int i = 0; i < 2000; ++i) {
    const double x = pos(rng), y = pos(rng), a = ang(rng);
    const std::vector<Point> pts{{float(x), float(y), 0.f, 0.f}, {float(x + far * std::cos(a)), float(y + far * std::sin(a)), 0.f, 0.f}};
    ASSERT_EQ(grid_clusters(pts, 0.15, 0.5, 1, 1000).size(), 2u);
  }
}

TEST(GridCluster, CellStatsAndSizeLimits)
{
  std::vector<Point> pts;
  for (int k = 0; k < 10; ++k) {pts.push_back({0.01f, 0.01f, 0.1f * k, 0.f});}  // one cell, 10 points
  auto cl = grid_clusters(pts, 0.15, 0.5, 1, 1000);
  ASSERT_EQ(cl.size(), 1u);
  ASSERT_EQ(cl[0].size(), 1u);
  EXPECT_EQ(cl[0][0].count, 10);
  EXPECT_FLOAT_EQ(cl[0][0].zmin, 0.0f);
  EXPECT_NEAR(cl[0][0].zmax, 0.9f, 1e-5);
  EXPECT_TRUE(grid_clusters(pts, 0.15, 0.5, 2, 1000).empty());  // 1 cell < min 2
}
