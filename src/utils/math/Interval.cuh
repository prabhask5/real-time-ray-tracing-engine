#pragma once

#ifdef USE_CUDA

#include "Utility.cuh"
#include <cmath>

// Represents a numeric interval [min, max].
struct CudaInterval {
  double min = +CUDA_INF;
  double max = -CUDA_INF;
};

// Constants — defined as `__constant__` if reused across kernels, or
// `__device__` if needed per thread.
__device__ __constant__ CudaInterval CUDA_EMPTY_INTERVAL = {+CUDA_INF,
                                                            -CUDA_INF};
__device__ __constant__ CudaInterval CUDA_UNIVERSE_INTERVAL = {-CUDA_INF,
                                                               +CUDA_INF};

__device__ inline CudaInterval cuda_make_interval(double min, double max) {
  return CudaInterval{min, max};
}

// Create the interval tightly enclosing the two input intervals.
__device__ inline CudaInterval cuda_make_union_interval(const CudaInterval &a,
                                                        const CudaInterval &b) {
  return CudaInterval{a.min <= b.min ? a.min : b.min,
                      a.max >= b.max ? a.max : b.max};
}

__device__ inline double cuda_interval_size(const CudaInterval &i) {
  return i.max - i.min;
}

__device__ inline bool cuda_interval_contains(const CudaInterval &i, double x) {
  return i.min <= x && x <= i.max;
}

__device__ inline bool cuda_interval_surrounds(const CudaInterval &i,
                                               double x) {
  return i.min < x && x < i.max;
}

__device__ inline double cuda_interval_clamp(const CudaInterval &i, double x) {
  return fmin(fmax(x, i.min), i.max); // Single branchless call.
}

// Make a new interval that is expanded by the delta value.
__device__ inline CudaInterval cuda_interval_expand(const CudaInterval &i,
                                                    double delta) {
  double padding = delta * 0.5;
  return CudaInterval{i.min - padding, i.max + padding};
}

__device__ inline void cuda_interval_set(CudaInterval &i, double new_min,
                                         double new_max) {
  i.min = new_min;
  i.max = new_max;
}

// GPU batch processing functions (implemented in .cu file).
void cuda_batch_clamp_values(double *d_values, const CudaInterval *d_intervals,
                             int count);
void cuda_batch_intersect_intervals(const CudaInterval *d_intervals1,
                                    const CudaInterval *d_intervals2,
                                    CudaInterval *d_result, int count);

#endif // USE_CUDA
