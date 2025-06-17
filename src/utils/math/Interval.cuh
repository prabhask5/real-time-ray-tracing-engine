#pragma once

#ifdef USE_CUDA

#include "Utility.cuh"

// Represents a numeric interval [min, max].
class Interval {
public:
  // Simulates an empty interval.
  __device__ Interval() : m_min(+INF), m_max(-INF) {}

  __device__ Interval(double min, double max) : m_min(min), m_max(max) {}

  // Create the interval tightly enclosing the two input intervals.
  __device__ Interval(const Interval &a, const Interval &b) {
    m_min = a.min() <= b.min() ? a.min() : b.min();
    m_max = a.max() >= b.max() ? a.max() : b.max();
  }

  // Getter const methods.

  __device__ double min() const { return m_min; }

  __device__ double max() const { return m_max; }

  __device__ double size() const { return m_max - m_min; }

  __device__ bool contains(double x) const { return m_min <= x && x <= m_max; }

  __device__ bool surrounds(double x) const { return m_min < x && x < m_max; }

  __device__ double clamp(double x) const {
    if (x < m_min)
      return m_min;
    if (x > m_max)
      return m_max;
    return x;
  }

  // Make a new interval that is expanded by the delta value.
  __device__ Interval expand(double delta) const {
    double padding = delta / 2;
    return Interval(m_min - padding, m_max + padding);
  }

  __device__ void set_interval(double min, double max) {
    m_min = min;
    m_max = max;
  }

private:
  double m_min;
  double m_max;
};

// Global constants representing common interval types.
__device__ static const Interval EMPTY_INTERVAL = Interval(+INF, -INF);
__device__ static const Interval UNIVERSE_INTERVAL = Interval(-INF, +INF);

// GPU batch processing functions (implemented in .cu file).
void cuda_batch_clamp_values(double *d_values, const Interval *d_intervals,
                             int count);
void cuda_batch_intersect_intervals(const Interval *d_intervals1,
                                    const Interval *d_intervals2,
                                    Interval *d_result, int count);

#endif // USE_CUDA
