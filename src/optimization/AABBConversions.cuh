#pragma once

#ifdef USE_CUDA

#include "../utils/math/IntervalConversions.cuh"
#include "../utils/math/Vec3Conversions.cuh"
#include "AABB.cuh"
#include "AABB.hpp"

// Convert CPU AABB to CUDA AABB POD struct.
inline CudaAABB cpu_to_cuda_aabb(const AABB &cpu_aabb) {
  return cuda_make_aabb(cpu_to_cuda_interval(cpu_aabb.x()),
                        cpu_to_cuda_interval(cpu_aabb.y()),
                        cpu_to_cuda_interval(cpu_aabb.z()));
}

#endif // USE_CUDA