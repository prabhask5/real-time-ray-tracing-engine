#pragma once

#ifdef USE_CUDA

#include "Interval.cuh"
#include "Interval.hpp"

// Convert CPU Interval to CUDA Interval POD struct.
inline CudaInterval cpu_to_cuda_interval(const Interval &cpu_interval) {
  return cuda_make_interval(cpu_interval.min(), cpu_interval.max());
}

#endif // USE_CUDA