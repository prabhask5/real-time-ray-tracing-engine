#pragma once

#ifdef USE_CUDA

#include "Utility.cuh"
#include <cmath>

// Represents a numeric interval [min, max].
struct CudaInterval {
  double min;
  double max;

  // Empty interval.
  __device__ CudaInterval() : min(+CUDA_INF), max(-CUDA_INF) {}

  __device__ CudaInterval(double _min, double _max) : min(_min), max(_max) {}

  // Create the interval tightly enclosing the two input intervals.
  __device__ CudaInterval(const CudaInterval &a, const CudaInterval &b) {
    min = a.min <= b.min ? a.min : b.min;
    max = a.max >= b.max ? a.max : b.max;
  }

  // Getter const methods.

  __device__ inline double size() { return max - min; }

  __device__ inline bool contains(double x) { return min <= x && x <= max; }

  __device__ inline bool surrounds(double x) { return min < x && x < max; }

  __device__ inline double clamp(double x) {
    return fmin(fmax(x, min), max); // Single branchless call.
  }

  // Make a new interval that is expanded by the delta value.
  __device__ inline CudaInterval expand(double delta) {
    double padding = delta * 0.5;
    return CudaInterval(min - padding, max + padding);
  }
};

// Constants — defined as `__constant__` if reused across kernels, or
// `__device__` if needed per thread.
__device__ __constant__ CudaInterval CUDA_EMPTY_INTERVAL = {+CUDA_INF,
                                                            -CUDA_INF};
__device__ __constant__ CudaInterval CUDA_UNIVERSE_INTERVAL = {-CUDA_INF,
                                                               +CUDA_INF};

// GPU batch processing functions (implemented in .cu file).
void cuda_batch_clamp_values(double *d_values, const CudaInterval *d_intervals,
                             int count);
void cuda_batch_intersect_intervals(const CudaInterval *d_intervals1,
                                    const CudaInterval *d_intervals2,
                                    CudaInterval *d_result, int count);

#endif // USE_CUDA
