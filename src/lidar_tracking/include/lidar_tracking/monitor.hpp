// Runtime monitor for track outputs. It sees only what a downstream consumer sees
// (track id + box per frame, plus the ego pose) and flags physically implausible
// changes between a track's consecutive reports. Thresholds are set a priori from
// physics, not tuned on data.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "lidar_tracking/tracker.hpp"
#include "lidar_tracking/types.hpp"

namespace lidar_tracking
{

enum MonitorFlag : uint8_t
{
  kFlagNone = 0,
  kFlagSpeed = 1,  // speed above max_speed_mps (any object)
  kFlagSmallSpeed = 2,  // small footprint moving faster than max_small_speed_mps
  kFlagSize = 4,  // BEV area changed by more than max_area_ratio between reports
};

struct MonitorParams
{
  double max_speed_mps{40.0};  // ~144 km/h: above urban traffic by a wide margin
  double small_area_m2{1.0};  // pedestrian-sized footprint
  double max_small_speed_mps{12.0};  // faster than a sprinting pedestrian / most cyclists
  double max_area_ratio{2.0};  // a rigid object does not double its footprint in one report
  double min_area_m2{0.05};  // ignore ratios of near-degenerate boxes
  double forget_after_s{1.0};  // drop state of tracks not seen for this long
};

struct MonitorCounts
{
  int64_t reports{0};
  int64_t speed{0};
  int64_t small_speed{0};
  int64_t size{0};
  int64_t any{0};
};

class TrackMonitor
{
public:
  explicit TrackMonitor(MonitorParams params = {});

  // One call per frame with that frame's track reports. Returns one flag mask per report.
  std::vector<uint8_t> check(const std::vector<TrackOutput> & tracks, const Pose2D & ego, double t_s);

  const MonitorCounts & counts() const {return counts_;}

private:
  struct State
  {
    double x{0.0};
    double y{0.0};
    double area{0.0};
    double t{0.0};
  };
  MonitorParams params_;
  std::unordered_map<int64_t, State> last_;
  MonitorCounts counts_;
};

}  // namespace lidar_tracking
