#pragma once

#ifdef USE_CUDA

#include "Utility.cuh"
#include <cmath>

// Represents a numeric interval [min, max].
struct CudaInterval {
  double min;
  double max;

  // Simulates empty interval.
  __device__ CudaInterval() : min(+CUDA_INF), max(-CUDA_INF) {}

  __host__ __device__ CudaInterval(double _min, double _max)
      : min(_min), max(_max) {}

  // Create the interval tightly enclosing the two input intervals.
  __device__ CudaInterval(const CudaInterval &a, const CudaInterval &b) {
    min = a.min <= b.min ? a.min : b.min;
    max = a.max >= b.max ? a.max : b.max;
  }

  // Getter const methods.

  __device__ inline double size() const { return max - min; }

  __device__ inline bool contains(double x) const {
    return min <= x && x <= max;
  }

  __device__ inline bool surrounds(double x) const {
    return min < x && x < max;
  }

  __device__ inline double clamp(double x) const {
    return fmin(fmax(x, min), max); // Single branchless call.
  }

  // Make a new interval that is expanded by the delta value.
  __device__ inline CudaInterval expand(double delta) const {
    double padding = delta * 0.5;
    return CudaInterval(min - padding, max + padding);
  }
};

// Constants — use inline functions to avoid initialization issues
__device__ inline CudaInterval cuda_empty_interval() {
  return CudaInterval(+CUDA_INF, -CUDA_INF);
}

__device__ inline CudaInterval cuda_universe_interval() {
  return CudaInterval(-CUDA_INF, +CUDA_INF);
}

// For compatibility, define macros.
#define CUDA_EMPTY_INTERVAL cuda_empty_interval()
#define CUDA_UNIVERSE_INTERVAL cuda_universe_interval()

#endif // USE_CUDA
