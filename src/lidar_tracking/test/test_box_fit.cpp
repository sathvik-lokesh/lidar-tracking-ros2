#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "lidar_tracking/box_fit.hpp"

using lidar_tracking::Point;
using lidar_tracking::fit_min_area_box;

namespace
{
// Points on the outline of an l x w rectangle centred at (cx, cy), rotated by yaw.
std::vector<Point> rect_outline(double cx, double cy, double l, double w, double yaw, double z0, double z1)
{
  std::vector<Point> pts;
  const double c = std::cos(yaw), s = std::sin(yaw);
  for (int i = 0; i <= 20; ++i) {
    const double u = -l / 2 + l * i / 20.0;
    for (double v : {-w / 2, w / 2}) {
      for (double z : {z0, z1}) {
        pts.push_back({static_cast<float>(cx + c * u - s * v), static_cast<float>(cy + s * u + c * v), static_cast<float>(z), 0.f});
      }
    }
  }
  for (int i = 0; i <= 10; ++i) {
    const double v = -w / 2 + w * i / 10.0;
    for (double u : {-l / 2, l / 2}) {
      pts.push_back({static_cast<float>(cx + c * u - s * v), static_cast<float>(cy + s * u + c * v), static_cast<float>(z0), 0.f});
    }
  }
  return pts;
}
}  // namespace

TEST(BoxFit, AxisAligned)
{
  const auto b = fit_min_area_box(rect_outline(10.0, -3.0, 4.0, 2.0, 0.0, -0.3, 1.2));
  EXPECT_NEAR(b.cx, 10.0, 1e-3);
  EXPECT_NEAR(b.cy, -3.0, 1e-3);
  EXPECT_NEAR(b.length, 4.0, 1e-3);
  EXPECT_NEAR(b.width, 2.0, 1e-3);
  EXPECT_NEAR(std::sin(b.yaw), 0.0, 1e-3);  // yaw 0 (box is symmetric)
  EXPECT_NEAR(b.cz, 0.45, 1e-3);
  EXPECT_NEAR(b.height, 1.5, 1e-3);
}

TEST(BoxFit, Rotated30Deg)
{
  const double yaw = 30.0 * M_PI / 180.0;
  const auto b = fit_min_area_box(rect_outline(-5.0, 7.0, 4.5, 1.8, yaw, 0.0, 1.0));
  EXPECT_NEAR(b.cx, -5.0, 1e-3);
  EXPECT_NEAR(b.cy, 7.0, 1e-3);
  EXPECT_NEAR(b.length, 4.5, 1e-3);
  EXPECT_NEAR(b.width, 1.8, 1e-3);
  EXPECT_NEAR(b.yaw, yaw, 1e-4);
}

TEST(BoxFit, LengthAlwaysLongerSideAndYawNormalised)
{
  const double yaw = 120.0 * M_PI / 180.0;  // same box as -60 deg
  const auto b = fit_min_area_box(rect_outline(0.0, 0.0, 5.0, 1.0, yaw, 0.0, 1.0));
  EXPECT_GE(b.length, b.width);
  EXPECT_GT(b.yaw, -M_PI / 2);
  EXPECT_LE(b.yaw, M_PI / 2);
  EXPECT_NEAR(b.yaw, -60.0 * M_PI / 180.0, 1e-4);
}

TEST(BoxFit, DegenerateClusters)
{
  const auto one = fit_min_area_box({{1.f, 2.f, 3.f, 0.f}});
  EXPECT_NEAR(one.cx, 1.0, 1e-6);
  EXPECT_NEAR(one.cy, 2.0, 1e-6);
  EXPECT_EQ(one.num_points, 1);

  const auto two = fit_min_area_box({{0.f, 0.f, 0.f, 0.f}, {0.f, 2.f, 0.f, 0.f}});
  EXPECT_NEAR(two.cx, 0.0, 1e-6);
  EXPECT_NEAR(two.cy, 1.0, 1e-6);
  EXPECT_NEAR(two.length, 2.0, 1e-6);
  EXPECT_NEAR(two.width, 0.0, 1e-6);

  // Collinear points must not crash and give a zero-width box along the line.
  const auto line = fit_min_area_box({{0.f, 0.f, 0.f, 0.f}, {1.f, 1.f, 0.f, 0.f}, {2.f, 2.f, 0.f, 0.f}});
  EXPECT_NEAR(line.length, 2.0 * std::sqrt(2.0), 1e-6);
  EXPECT_NEAR(line.width, 0.0, 1e-6);
}
