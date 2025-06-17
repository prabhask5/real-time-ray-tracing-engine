#pragma once

#ifdef USE_CUDA

#include "Interval.cuh"

// Adds a displacement to an interval (shifting the entire interval).
__device__ inline Interval operator+(const Interval &ival,
                                     double displacement) {
  return Interval(ival.min() + displacement, ival.max() + displacement);
}

// Allows addition in both operand orders.
__device__ inline Interval operator+(double displacement,
                                     const Interval &ival) {
  return ival + displacement;
}

#endif // USE_CUDA
