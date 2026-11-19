#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "lidar_tracking/hungarian.hpp"
#include "lidar_tracking/tracker.hpp"

using namespace lidar_tracking;

namespace
{
double brute_force_min(const std::vector<double> & c, int rows, int cols)
{
  // Assign every row of the smaller side; permutations of the larger side.
  const bool tall = rows > cols;
  const int n = tall ? cols : rows, m = tall ? rows : cols;
  std::vector<int> perm(m);
  std::iota(perm.begin(), perm.end(), 0);
  double best = std::numeric_limits<double>::max();
  do {
    double s = 0;
    for (int i = 0; i < n; ++i) {s += tall ? c[perm[i] * cols + i] : c[i * cols + perm[i]];}
    best = std::min(best, s);
  } while (std::next_permutation(perm.begin(), perm.end()));
  return best;
}

Box box_at(double x, double y)
{
  Box b;
  b.cx = x; b.cy = y; b.length = 4.0; b.width = 2.0; b.height = 1.5;
  return b;
}
}  // namespace

TEST(Hungarian, MatchesBruteForceOnRandomMatrices)
{
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(0.0, 10.0);
  for (int trial = 0; trial < 300; ++trial) {
    const int rows = 1 + trial % 6, cols = 1 + (trial / 6) % 6;
    std::vector<double> c(rows * cols);
    for (auto & v : c) {v = u(rng);}
    const auto a = hungarian(c, rows, cols);
    double s = 0;
    std::set<int> used;
    int assigned = 0;
    for (int i = 0; i < rows; ++i) {
      if (a[i] < 0) {continue;}
      ++assigned;
      EXPECT_TRUE(used.insert(a[i]).second) << "column used twice";
      s += c[i * cols + a[i]];
    }
    EXPECT_EQ(assigned, std::min(rows, cols));
    EXPECT_NEAR(s, brute_force_min(c, rows, cols), 1e-9) << rows << "x" << cols;
  }
}

TEST(Hungarian, EmptyInputs)
{
  EXPECT_TRUE(hungarian({}, 0, 3).empty());
  EXPECT_EQ(hungarian({}, 2, 0), std::vector<int>({-1, -1}));
}

TEST(Frames, EgoMapRoundTrip)
{
  const Pose2D ego{100.0, -50.0, 1.1};
  const Box b = box_at(12.0, -3.0);
  const Box r = map_to_ego(ego_to_map(b, ego), ego);
  EXPECT_NEAR(r.cx, b.cx, 1e-9);
  EXPECT_NEAR(r.cy, b.cy, 1e-9);
  EXPECT_NEAR(r.yaw, b.yaw, 1e-9);
}

TEST(Tracker, StaticObjectKeepsIdWhileEgoDrives)
{
  // Ego drives at 10 m/s with heading 30 deg past a parked car at map (20, 15).
  for (auto mode : {TrackerParams::Mode::Nearest, TrackerParams::Mode::Kalman}) {
    TrackerParams p;
    p.mode = mode;
    Tracker tr(p);
    std::set<int64_t> ids;
    const double yaw = M_PI / 6;
    for (int k = 0; k < 30; ++k) {
      const Pose2D ego{k * 1.0 * std::cos(yaw), k * 1.0 * std::sin(yaw), yaw};
      const Box det = map_to_ego(box_at(20.0, 15.0), ego);
      const auto out = tr.update({det}, ego, k * 0.1);
      ASSERT_EQ(out.size(), 1u);
      ids.insert(out[0].id);
      EXPECT_NEAR(out[0].box.cx, det.cx, 0.05);
      EXPECT_NEAR(out[0].box.cy, det.cy, 0.05);
    }
    EXPECT_EQ(ids.size(), 1u);
  }
}

TEST(Tracker, KalmanKeepsIdsThroughCloseOncomingPass)
{
  // Two cars in opposite lanes 1.2 m apart at 10 m/s each pass each other.
  TrackerParams p;
  p.mode = TrackerParams::Mode::Kalman;
  Tracker tr(p);
  const Pose2D ego{};
  int64_t id_a = -1, id_b = -1;
  for (int k = 0; k < 40; ++k) {
    const double t = k * 0.1;
    const Box a = box_at(-20.0 + 10.0 * t, 0.6), b = box_at(20.0 - 10.0 * t, -0.6);
    const auto out = tr.update({a, b}, ego, t);
    ASSERT_EQ(out.size(), 2u);
    for (const auto & o : out) {
      const bool is_a = std::abs(o.box.cy - 0.6) < 0.3;
      int64_t & id = is_a ? id_a : id_b;
      if (id < 0) {id = o.id;}
      EXPECT_EQ(o.id, id) << "id switch at frame " << k;
    }
  }
  EXPECT_NE(id_a, id_b);
}

TEST(Tracker, MinHitsHidesOneFrameClutterAndMaxMissesDeletes)
{
  TrackerParams p;
  p.mode = TrackerParams::Mode::Kalman;
  p.min_hits = 3;
  p.max_misses = 2;
  Tracker tr(p);
  const Pose2D ego{};
  // Frame 0: a real object plus a one-frame clutter blob far away.
  EXPECT_TRUE(tr.update({box_at(10, 0), box_at(-30, 30)}, ego, 0.0).empty());
  EXPECT_TRUE(tr.update({box_at(10, 0)}, ego, 0.1).empty());  // 2 hits
  auto out = tr.update({box_at(10, 0)}, ego, 0.2);  // 3 hits -> reported
  ASSERT_EQ(out.size(), 1u);
  const int64_t id = out[0].id;
  // Object vanishes for 2 frames (kept), comes back: same id.
  EXPECT_TRUE(tr.update({}, ego, 0.3).empty());
  EXPECT_TRUE(tr.update({}, ego, 0.4).empty());
  out = tr.update({box_at(10, 0)}, ego, 0.5);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].id, id);
  // Gone for 3 frames: deleted, a new detection becomes a new (unconfirmed) track.
  for (double t : {0.6, 0.7, 0.8}) {tr.update({}, ego, t);}
  EXPECT_EQ(tr.num_live_tracks(), 0u);
  EXPECT_TRUE(tr.update({box_at(10, 0)}, ego, 0.9).empty());
}

TEST(Tracker, NoneModeGivesFreshIds)
{
  TrackerParams p;
  p.mode = TrackerParams::Mode::None;
  Tracker tr(p);
  const auto a = tr.update({box_at(1, 1)}, {}, 0.0);
  const auto b = tr.update({box_at(1, 1)}, {}, 0.1);
  ASSERT_EQ(a.size(), 1u);
  ASSERT_EQ(b.size(), 1u);
  EXPECT_NE(a[0].id, b[0].id);
}
