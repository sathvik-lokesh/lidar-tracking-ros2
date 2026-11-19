#include "lidar_tracking/tracker.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <tuple>

#include "lidar_tracking/hungarian.hpp"

namespace lidar_tracking
{

namespace
{
using Vec4 = Eigen::Matrix<double, 4, 1>;
using Mat4 = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
constexpr double kInitVelStd = 10.0;  // m/s: a new track's velocity is unknown
constexpr double kForbidden = 1e6;
}  // namespace

Box ego_to_map(const Box & b, const Pose2D & ego)
{
  Box o = b;
  const double c = std::cos(ego.yaw), s = std::sin(ego.yaw);
  o.cx = ego.x + c * b.cx - s * b.cy;
  o.cy = ego.y + s * b.cx + c * b.cy;
  o.yaw = b.yaw + ego.yaw;
  return o;
}

Box map_to_ego(const Box & b, const Pose2D & ego)
{
  Box o = b;
  const double c = std::cos(ego.yaw), s = std::sin(ego.yaw);
  const double dx = b.cx - ego.x, dy = b.cy - ego.y;
  o.cx = c * dx + s * dy;
  o.cy = -s * dx + c * dy;
  o.yaw = b.yaw - ego.yaw;
  return o;
}

Tracker::Tracker(TrackerParams params)
: params_(params) {}

Tracker::Track Tracker::new_track(const Box & det_map)
{
  Track t;
  t.id = next_id_++;
  t.x = {det_map.cx, det_map.cy, 0.0, 0.0};
  Eigen::Map<Mat4> P(t.P.data());
  const double r2 = params_.meas_std_m * params_.meas_std_m;
  P = Vec4(r2, r2, kInitVelStd * kInitVelStd, kInitVelStd * kInitVelStd).asDiagonal();
  t.last_box = det_map;
  t.hits = 1;
  return t;
}

void Tracker::predict(Track & t, double dt) const
{
  if (dt <= 0.0) {return;}
  Eigen::Map<Vec4> x(t.x.data());
  Eigen::Map<Mat4> P(t.P.data());
  Mat4 F = Mat4::Identity();
  F(0, 2) = dt;
  F(1, 3) = dt;
  // Discrete white-noise acceleration model, per axis.
  const double q = params_.accel_std * params_.accel_std;
  const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
  Mat4 Q = Mat4::Zero();
  Q(0, 0) = Q(1, 1) = q * dt4 / 4.0;
  Q(0, 2) = Q(2, 0) = Q(1, 3) = Q(3, 1) = q * dt3 / 2.0;
  Q(2, 2) = Q(3, 3) = q * dt2;
  x = F * x;
  P = F * P * F.transpose() + Q;
}

void Tracker::correct(Track & t, double zx, double zy) const
{
  Eigen::Map<Vec4> x(t.x.data());
  Eigen::Map<Mat4> P(t.P.data());
  Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
  H(0, 0) = H(1, 1) = 1.0;
  const double r2 = params_.meas_std_m * params_.meas_std_m;
  const Eigen::Matrix2d S = H * P * H.transpose() + r2 * Eigen::Matrix2d::Identity();
  const Eigen::Matrix<double, 4, 2> K = P * H.transpose() * S.inverse();
  x += K * (Eigen::Vector2d(zx, zy) - H * x);
  // Joseph form keeps P symmetric positive definite.
  const Mat4 I_KH = Mat4::Identity() - K * H;
  P = I_KH * P * I_KH.transpose() + K * (r2 * Eigen::Matrix2d::Identity()) * K.transpose();
}

std::vector<TrackOutput> Tracker::update(const std::vector<Box> & detections, const Pose2D & ego, double t_s)
{
  std::vector<TrackOutput> out;
  if (params_.mode == TrackerParams::Mode::None) {
    for (const auto & d : detections) {out.push_back({d, next_id_++, 1});}
    return out;
  }

  const double dt = has_last_t_ ? t_s - last_t_ : 0.0;
  last_t_ = t_s;
  has_last_t_ = true;

  std::vector<Box> dets;
  dets.reserve(detections.size());
  for (const auto & d : detections) {dets.push_back(ego_to_map(d, ego));}

  const bool kalman = params_.mode == TrackerParams::Mode::Kalman;
  if (kalman) {
    for (auto & t : tracks_) {predict(t, dt);}
  }

  // Track positions used for association.
  const int nt = static_cast<int>(tracks_.size());
  const int nd = static_cast<int>(dets.size());
  std::vector<double> cost(static_cast<size_t>(nt) * nd, kForbidden);
  for (int i = 0; i < nt; ++i) {
    const double tx = kalman ? tracks_[i].x[0] : tracks_[i].last_box.cx;
    const double ty = kalman ? tracks_[i].x[1] : tracks_[i].last_box.cy;
    for (int j = 0; j < nd; ++j) {
      const double d = std::hypot(dets[j].cx - tx, dets[j].cy - ty);
      if (d <= params_.gate_m) {cost[i * nd + j] = d;}
    }
  }

  std::vector<int> track_to_det(nt, -1);
  if (kalman) {
    track_to_det = hungarian(cost, nt, nd);
    for (int i = 0; i < nt; ++i) {
      if (track_to_det[i] >= 0 && cost[i * nd + track_to_det[i]] >= kForbidden) {track_to_det[i] = -1;}
    }
  } else {
    // Greedy: take the globally closest gated pair, repeat.
    std::vector<std::tuple<double, int, int>> pairs;
    for (int i = 0; i < nt; ++i) {
      for (int j = 0; j < nd; ++j) {
        if (cost[i * nd + j] < kForbidden) {pairs.emplace_back(cost[i * nd + j], i, j);}
      }
    }
    std::sort(pairs.begin(), pairs.end());
    std::vector<char> det_used(nd, 0);
    for (const auto & [c, i, j] : pairs) {
      if (track_to_det[i] < 0 && !det_used[j]) {track_to_det[i] = j; det_used[j] = 1;}
    }
  }

  std::vector<char> det_matched(nd, 0);
  std::vector<Track> kept;
  kept.reserve(tracks_.size() + dets.size());
  for (int i = 0; i < nt; ++i) {
    Track & t = tracks_[i];
    const int j = track_to_det[i];
    if (j >= 0) {
      det_matched[j] = 1;
      t.hits += 1;
      t.misses = 0;
      t.last_box = dets[j];
      if (kalman) {correct(t, dets[j].cx, dets[j].cy);}
      if (t.hits >= params_.min_hits) {
        Box b = dets[j];
        if (kalman && params_.output_filtered) {b.cx = t.x[0]; b.cy = t.x[1];}
        out.push_back({map_to_ego(b, ego), t.id, t.hits});
      }
      kept.push_back(t);
    } else if (++t.misses <= params_.max_misses) {
      kept.push_back(t);
    }
  }
  for (int j = 0; j < nd; ++j) {
    if (det_matched[j]) {continue;}
    Track t = new_track(dets[j]);
    if (t.hits >= params_.min_hits) {out.push_back({map_to_ego(dets[j], ego), t.id, t.hits});}
    kept.push_back(t);
  }
  tracks_ = std::move(kept);
  return out;
}

}  // namespace lidar_tracking
