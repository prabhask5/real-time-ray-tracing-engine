#pragma once

#ifdef USE_CUDA

#include "Utility.cuh"
#include <cmath>
#include <iomanip>
#include <sstream>

// POD struct representing a numeric interval [min, max].
struct CudaInterval {
  double min;
  double max;
};

// Interval initialization functions.
__host__ __device__ inline CudaInterval cuda_make_interval() {
  CudaInterval interval;
  interval.min = +CUDA_INF;
  interval.max = -CUDA_INF;
  return interval;
}

__host__ __device__ inline CudaInterval cuda_make_interval(double min,
                                                           double max) {
  CudaInterval interval;
  interval.min = min;
  interval.max = max;
  return interval;
}

// Create the interval tightly enclosing the two input intervals.
__host__ __device__ inline CudaInterval
cuda_make_interval(const CudaInterval &a, const CudaInterval &b) {
  CudaInterval interval;
  interval.min = a.min <= b.min ? a.min : b.min;
  interval.max = a.max >= b.max ? a.max : b.max;
  return interval;
}

// Interval utility functions.
__host__ __device__ inline double
cuda_interval_size(const CudaInterval &interval) {
  return interval.max - interval.min;
}

__host__ __device__ inline bool
cuda_interval_contains(const CudaInterval &interval, double x) {
  return interval.min <= x && x <= interval.max;
}

__host__ __device__ inline bool
cuda_interval_surrounds(const CudaInterval &interval, double x) {
  return interval.min < x && x < interval.max;
}

__host__ __device__ inline double
cuda_interval_clamp(const CudaInterval &interval, double x) {
  return fmin(fmax(x, interval.min), interval.max); // Single branchless call.
}

// Make a new interval that is expanded by the delta value.
__host__ __device__ inline CudaInterval
cuda_interval_expand(const CudaInterval &interval, double delta) {
  double padding = delta * 0.5;
  return cuda_make_interval(interval.min - padding, interval.max + padding);
}

// Constants — use inline functions to avoid initialization issues.
__host__ __device__ inline CudaInterval cuda_empty_interval() {
  return cuda_make_interval(+CUDA_INF, -CUDA_INF);
}

__host__ __device__ inline CudaInterval cuda_universe_interval() {
  return cuda_make_interval(-CUDA_INF, +CUDA_INF);
}

// For compatibility, define macros.
#define CUDA_EMPTY_INTERVAL cuda_empty_interval()
#define CUDA_UNIVERSE_INTERVAL cuda_universe_interval()

// JSON serialization function for CudaInterval.
inline std::string cuda_json_interval(const CudaInterval &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaInterval\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"min\":" << obj.min << ",";
  oss << "\"max\":" << obj.max;
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA
