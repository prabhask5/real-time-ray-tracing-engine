#pragma once

#ifdef USE_CUDA

#include "ONB.cuh"
#include "ONB.hpp"
#include "Vec3Conversions.cuh"

// Convert CPU ONB to CUDA ONB POD struct.
inline CudaONB cpu_to_cuda_onb(const ONB &cpu_onb) {
  return cuda_make_onb(cpu_to_cuda_vec3(cpu_onb.w()));
}

#endif // USE_CUDA