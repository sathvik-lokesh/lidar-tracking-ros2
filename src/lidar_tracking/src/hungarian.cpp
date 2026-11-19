#include "lidar_tracking/hungarian.hpp"

#include <limits>

namespace lidar_tracking
{

// Classic potentials formulation (e-maxx). Works on an n x m matrix with n <= m;
// a taller matrix is solved transposed.
std::vector<int> hungarian(const std::vector<double> & cost, int rows, int cols)
{
  std::vector<int> result(rows, -1);
  if (rows == 0 || cols == 0) {
    return result;
  }
  const bool transposed = rows > cols;
  const int n = transposed ? cols : rows;
  const int m = transposed ? rows : cols;
  auto a = [&](int i, int j) {  // 1-based
      return transposed ? cost[(j - 1) * cols + (i - 1)] : cost[(i - 1) * cols + (j - 1)];
    };

  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
  std::vector<int> p(m + 1, 0), way(m + 1, 0);
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(m + 1, inf);
    std::vector<char> used(m + 1, 0);
    do {
      used[j0] = 1;
      const int i0 = p[j0];
      double delta = inf;
      int j1 = 0;
      for (int j = 1; j <= m; ++j) {
        if (used[j]) {continue;}
        const double cur = a(i0, j) - u[i0] - v[j];
        if (cur < minv[j]) {minv[j] = cur; way[j] = j0;}
        if (minv[j] < delta) {delta = minv[j]; j1 = j;}
      }
      for (int j = 0; j <= m; ++j) {
        if (used[j]) {u[p[j]] += delta; v[j] -= delta;} else {minv[j] -= delta;}
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  for (int j = 1; j <= m; ++j) {
    if (p[j] == 0) {continue;}
    if (transposed) {
      result[j - 1] = p[j] - 1;  // row (j-1) of the original matrix -> column p[j]-1
    } else {
      result[p[j] - 1] = j - 1;
    }
  }
  return result;
}

}  // namespace lidar_tracking
