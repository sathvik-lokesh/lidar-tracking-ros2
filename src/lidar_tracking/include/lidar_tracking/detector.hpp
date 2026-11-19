// Class-agnostic lidar object detector: crop -> ground removal -> voxel grid ->
// Euclidean clustering -> BEV min-area boxes. PCL stays in detector.cpp (pimpl),
// so including this header does not pull PCL into every translation unit.
#pragma once

#include <memory>
#include <vector>

#include "lidar_tracking/types.hpp"

namespace lidar_tracking
{

class Detector
{
public:
  explicit Detector(DetectorParams params = {});
  ~Detector();
  Detector(const Detector &) = delete;
  Detector & operator=(const Detector &) = delete;

  // Points in the ego frame (z up). Returns boxes in the same frame.
  std::vector<Box> detect(const std::vector<Point> & cloud, StageTimings * timings = nullptr);

  // Exposed for tests and the ground-removal ablation.
  std::vector<Point> remove_ground(const std::vector<Point> & cloud) const;

  const DetectorParams & params() const {return params_;}

private:
  struct Impl;
  DetectorParams params_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lidar_tracking
