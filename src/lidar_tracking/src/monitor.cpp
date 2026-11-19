#include "lidar_tracking/monitor.hpp"

#include <algorithm>
#include <cmath>

namespace lidar_tracking
{

TrackMonitor::TrackMonitor(MonitorParams params)
: params_(params) {}

std::vector<uint8_t> TrackMonitor::check(
  const std::vector<TrackOutput> & tracks, const Pose2D & ego, double t_s)
{
  std::vector<uint8_t> flags(tracks.size(), kFlagNone);
  for (size_t i = 0; i < tracks.size(); ++i) {
    const Box b = ego_to_map(tracks[i].box, ego);
    const double area = b.length * b.width;
    auto it = last_.find(tracks[i].id);
    if (it != last_.end()) {
      const State & s = it->second;
      const double dt = t_s - s.t;
      if (dt > 1e-3) {
        const double speed = std::hypot(b.cx - s.x, b.cy - s.y) / dt;
        if (speed > params_.max_speed_mps) {flags[i] |= kFlagSpeed;}
        if (std::max(area, s.area) < params_.small_area_m2 && speed > params_.max_small_speed_mps) {
          flags[i] |= kFlagSmallSpeed;
        }
      }
      if (area > params_.min_area_m2 && s.area > params_.min_area_m2) {
        const double ratio = std::max(area, s.area) / std::min(area, s.area);
        if (ratio > params_.max_area_ratio) {flags[i] |= kFlagSize;}
      }
    }
    last_[tracks[i].id] = {b.cx, b.cy, area, t_s};

    counts_.reports += 1;
    counts_.speed += (flags[i] & kFlagSpeed) != 0;
    counts_.small_speed += (flags[i] & kFlagSmallSpeed) != 0;
    counts_.size += (flags[i] & kFlagSize) != 0;
    counts_.any += flags[i] != kFlagNone;
  }
  for (auto it = last_.begin(); it != last_.end(); ) {
    it = (t_s - it->second.t > params_.forget_after_s) ? last_.erase(it) : std::next(it);
  }
  return flags;
}

}  // namespace lidar_tracking
