#pragma once

#ifdef USE_CUDA

#include "Interval.cuh"
#include "Interval.hpp"

// Convert CPU Interval to CUDA Interval
inline CudaInterval cpu_to_cuda_interval(const Interval &cpu_interval) {
  return CudaInterval(cpu_interval.min(), cpu_interval.max());
}

// Convert CUDA Interval to CPU Interval
inline Interval cuda_to_cpu_interval(const CudaInterval &cuda_interval) {
  return Interval(cuda_interval.min, cuda_interval.max);
}

// Batch conversion functions for performance
void batch_cpu_to_cuda_interval(const Interval *cpu_intervals,
                                CudaInterval *cuda_intervals, int count);
void batch_cuda_to_cpu_interval(const CudaInterval *cuda_intervals,
                                Interval *cpu_intervals, int count);

#endif // USE_CUDA