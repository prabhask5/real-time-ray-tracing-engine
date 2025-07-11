#pragma once

#ifdef USE_CUDA

#include "Interval.cuh"

// Adds a displacement to an interval (shifting the entire interval).
__host__ __device__ inline CudaInterval operator+(const CudaInterval &ival,
                                                  double displacement) {
  return cuda_make_interval(ival.min + displacement, ival.max + displacement);
}

// Allows addition in both operand orders.
__host__ __device__ inline CudaInterval operator+(double displacement,
                                                  const CudaInterval &ival) {
  return ival + displacement;
}

#endif // USE_CUDA
