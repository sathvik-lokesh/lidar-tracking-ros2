#include <gtest/gtest.h>

#include <cmath>

#include "lidar_tracking/monitor.hpp"

using namespace lidar_tracking;

namespace
{
TrackOutput rep(int64_t id, double x, double y, double l, double w)
{
  TrackOutput o;
  o.id = id;
  o.box.cx = x; o.box.cy = y; o.box.length = l; o.box.width = w; o.box.height = 1.5;
  return o;
}
}  // namespace

TEST(Monitor, PlausibleCarIsNotFlagged)
{
  TrackMonitor m;
  for (int k = 0; k < 20; ++k) {  // 15 m/s car
    const auto f = m.check({rep(1, 1.5 * k, 0, 4.5, 1.9)}, {}, 0.1 * k);
    EXPECT_EQ(f[0], kFlagNone) << k;
  }
  EXPECT_EQ(m.counts().any, 0);
  EXPECT_EQ(m.counts().reports, 20);
}

TEST(Monitor, TeleportingTrackIsFlagged)
{
  TrackMonitor m;
  m.check({rep(1, 0, 0, 4.5, 1.9)}, {}, 0.0);
  const auto f = m.check({rep(1, 5, 0, 4.5, 1.9)}, {}, 0.1);  // 50 m/s
  EXPECT_TRUE(f[0] & kFlagSpeed);
}

TEST(Monitor, FastSmallObjectIsFlaggedButFastCarIsNot)
{
  TrackMonitor m;
  m.check({rep(1, 0, 0, 0.6, 0.6), rep(2, 0, 10, 4.5, 1.9)}, {}, 0.0);
  const auto f = m.check({rep(1, 2, 0, 0.6, 0.6), rep(2, 2, 10, 4.5, 1.9)}, {}, 0.1);  // both 20 m/s
  EXPECT_TRUE(f[0] & kFlagSmallSpeed);
  EXPECT_EQ(f[1], kFlagNone);
}

TEST(Monitor, SizeJumpIsFlagged)
{
  TrackMonitor m;
  m.check({rep(1, 0, 0, 0.6, 0.6)}, {}, 0.0);
  const auto f = m.check({rep(1, 0.1, 0, 4.5, 1.9)}, {}, 0.1);  // ped box -> car box
  EXPECT_TRUE(f[0] & kFlagSize);
}

TEST(Monitor, EgoMotionAloneIsNotFlagged)
{
  // A parked car seen from an ego driving 20 m/s: in the ego frame it moves fast,
  // in the map frame it is still.
  TrackMonitor m;
  for (int k = 0; k < 10; ++k) {
    const Pose2D ego{2.0 * k, 0.0, 0.0};
    const auto f = m.check({rep(1, 30.0 - 2.0 * k, 3, 4.5, 1.9)}, ego, 0.1 * k);
    EXPECT_EQ(f[0], kFlagNone);
  }
}

TEST(Monitor, ForgetsStaleTracks)
{
  TrackMonitor m;
  m.check({rep(1, 0, 0, 4.5, 1.9)}, {}, 0.0);
  m.check({}, {}, 2.0);  // > forget_after_s
  const auto f = m.check({rep(1, 100, 0, 4.5, 1.9)}, {}, 2.1);  // same id far away: fresh state
  EXPECT_EQ(f[0], kFlagNone);
}
