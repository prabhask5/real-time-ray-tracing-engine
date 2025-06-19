#pragma once

#ifdef USE_CUDA

#include "ONB.cuh"
#include "ONB.hpp"
#include "Vec3Conversions.cuh"

// Convert CPU ONB to CUDA ONB.
inline CudaONB cpu_to_cuda_onb(const ONB &cpu_onb) {
  return CudaONB(cpu_to_cuda_vec3(cpu_onb.w()));
}

// Convert CUDA ONB to CPU ONB  .
inline ONB cuda_to_cpu_onb(const CudaONB &cuda_onb) {
  return ONB(cuda_to_cpu_vec3(cuda_onb.w()));
}

// Batch conversion functions for performance.
void batch_cpu_to_cuda_onb(const ONB *cpu_onbs, CudaONB *cuda_onbs, int count);
void batch_cuda_to_cpu_onb(const CudaONB *cuda_onbs, ONB *cpu_onbs, int count);

#endif // USE_CUDA