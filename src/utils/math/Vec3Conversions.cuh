#pragma once

#ifdef USE_CUDA

#include "Vec3.cuh"
#include "Vec3.hpp"

// Convert CPU Vec3 to CUDA Vec3 POD struct.
inline CudaVec3 cpu_to_cuda_vec3(const Vec3 &cpu_vec) {
  return cuda_make_vec3(cpu_vec.x(), cpu_vec.y(), cpu_vec.z());
}

// Convert CUDA Vec3 POD struct to CPU Vec3.
inline Vec3 cuda_to_cpu_vec3(const CudaVec3 &cuda_vec) {
  return Vec3(cuda_vec.x, cuda_vec.y, cuda_vec.z);
}

#endif // USE_CUDA