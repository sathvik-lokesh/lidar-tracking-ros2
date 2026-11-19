// Multi-object tracker on BEV box centres, run in a world-fixed (map) frame so the
// ego's own motion does not look like object motion.
//
// Modes, as an ablation ladder:
//   None     every detection gets a fresh id (no tracking)
//   Nearest  greedy nearest-neighbour association, a track sits at its last detection
//   Kalman   constant-velocity Kalman filter [x y vx vy] + Hungarian assignment
// Nearest and Kalman share the track lifecycle: a track is output once it has
// min_hits matched detections, and is deleted after max_misses frames unmatched.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "lidar_tracking/types.hpp"

namespace lidar_tracking
{

struct Pose2D
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

struct TrackerParams
{
  enum class Mode { None, Nearest, Kalman };
  Mode mode{Mode::Kalman};
  double gate_m{2.0};  // max BEV centre distance for an association
  int min_hits{1};  // matched detections before a track is output
  int max_misses{2};  // frames a track may go unmatched before deletion
  double accel_std{3.0};  // Kalman process noise: white acceleration, m/s^2
  double meas_std_m{0.3};  // Kalman measurement noise on x and y, m
  bool output_filtered{true};  // Kalman: output the filtered centre (else the raw detection)
};

struct TrackOutput
{
  Box box;  // ego frame, like the input detections
  int64_t id{0};
  int hits{0};
};

class Tracker
{
public:
  explicit Tracker(TrackerParams params = {});

  // detections are in the ego frame; ego is the ego pose in the map frame at time t_s.
  // Returns the tracks to report this frame (matched this frame and confirmed).
  std::vector<TrackOutput> update(const std::vector<Box> & detections, const Pose2D & ego, double t_s);

  size_t num_live_tracks() const {return tracks_.size();}

private:
  struct Track
  {
    int64_t id{0};
    std::array<double, 4> x{};  // map frame: x, y, vx, vy
    std::array<double, 16> P{};  // row-major 4x4 covariance
    Box last_box;  // last matched detection, map frame
    int hits{0};
    int misses{0};
  };

  void predict(Track & t, double dt) const;
  void correct(Track & t, double zx, double zy) const;
  Track new_track(const Box & det_map);

  TrackerParams params_;
  std::vector<Track> tracks_;
  int64_t next_id_{0};
  double last_t_{0.0};
  bool has_last_t_{false};
};

// Frame helpers (2D rigid transforms of box centre + yaw; z and size pass through).
Box ego_to_map(const Box & b, const Pose2D & ego);
Box map_to_ego(const Box & b, const Pose2D & ego);

}  // namespace lidar_tracking
