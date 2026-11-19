#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "lidar_tracking/detector.hpp"

using lidar_tracking::Box;
using lidar_tracking::Detector;
using lidar_tracking::Point;

namespace
{
// Sloped road: z = -0.4 + 0.05 * x (5 % grade), sampled every 0.2 m out to 40 m.
float ground_z(float x) {return -0.4f + 0.05f * x;}

std::vector<Point> sloped_ground()
{
  std::vector<Point> pts;
  for (float x = -40.f; x <= 40.f; x += 0.2f) {
    for (float y = -40.f; y <= 40.f; y += 0.2f) {
      if (x * x + y * y < 40.f * 40.f) {pts.push_back({x, y, ground_z(x), 0.f});}
    }
  }
  return pts;
}

// Filled axis-aligned block standing on the ground at (cx, cy).
void add_block(std::vector<Point> & pts, float cx, float cy, float l, float w, float h)
{
  const float g = ground_z(cx);
  for (float x = cx - l / 2; x <= cx + l / 2 + 1e-4f; x += 0.1f) {
    for (float y = cy - w / 2; y <= cy + w / 2 + 1e-4f; y += 0.1f) {
      for (float z = g + 0.05f; z <= g + h; z += 0.1f) {pts.push_back({x, y, z, 0.f});}
    }
  }
}

const Box * nearest(const std::vector<Box> & boxes, double x, double y)
{
  const Box * best = nullptr;
  double best_d = 1e9;
  for (const auto & b : boxes) {
    const double d = std::hypot(b.cx - x, b.cy - y);
    if (d < best_d) {best_d = d; best = &b;}
  }
  return best;
}
}  // namespace

TEST(Detector, GroundOnlyGivesNoBoxes)
{
  Detector det;
  EXPECT_TRUE(det.remove_ground(sloped_ground()).empty());
  EXPECT_TRUE(det.detect(sloped_ground()).empty());
}

TEST(Detector, FindsCarAndPedestrianOnSlope)
{
  auto pts = sloped_ground();
  add_block(pts, 15.f, 2.f, 4.4f, 1.8f, 1.5f);  // car uphill
  add_block(pts, -10.f, -6.f, 0.5f, 0.5f, 1.7f);  // pedestrian downhill
  Detector det;
  lidar_tracking::StageTimings t;
  const auto boxes = det.detect(pts, &t);
  ASSERT_EQ(boxes.size(), 2u);
  EXPECT_EQ(t.n_boxes, 2);

  const Box * car = nearest(boxes, 15.0, 2.0);
  ASSERT_NE(car, nullptr);
  EXPECT_LT(std::hypot(car->cx - 15.0, car->cy - 2.0), 0.15);
  EXPECT_NEAR(car->length, 4.4, 0.25);
  EXPECT_NEAR(car->width, 1.8, 0.25);

  const Box * ped = nearest(boxes, -10.0, -6.0);
  ASSERT_NE(ped, nullptr);
  EXPECT_LT(std::hypot(ped->cx + 10.0, ped->cy + 6.0), 0.15);
}

TEST(Detector, GridMethodFindsCarAndPedestrianOnSlope)
{
  auto pts = sloped_ground();
  add_block(pts, 15.f, 2.f, 4.4f, 1.8f, 1.5f);
  add_block(pts, -10.f, -6.f, 0.5f, 0.5f, 1.7f);
  lidar_tracking::DetectorParams p;
  p.cluster_method = 1;
  Detector det(p);
  const auto boxes = det.detect(pts);
  ASSERT_EQ(boxes.size(), 2u);
  const Box * car = nearest(boxes, 15.0, 2.0);
  EXPECT_LT(std::hypot(car->cx - 15.0, car->cy - 2.0), 0.15);
  EXPECT_NEAR(car->length, 4.4, 0.3);
  const Box * ped = nearest(boxes, -10.0, -6.0);
  EXPECT_LT(std::hypot(ped->cx + 10.0, ped->cy + 6.0), 0.15);
}

TEST(Detector, RangeCropDropsFarObjects)
{
  auto pts = sloped_ground();
  add_block(pts, 30.f, 0.f, 4.f, 2.f, 1.5f);
  lidar_tracking::DetectorParams p;
  p.max_range_m = 20.0;
  Detector det(p);
  EXPECT_TRUE(det.detect(pts).empty());
}

TEST(Detector, OversizedClusterRejected)
{
  auto pts = sloped_ground();
  add_block(pts, 0.f, 15.f, 30.f, 0.6f, 2.5f);  // 30 m wall
  Detector det;
  EXPECT_TRUE(det.detect(pts).empty());
}
