#pragma once

#ifdef USE_CUDA

#include "../utils/math/IntervalConversions.cuh"
#include "../utils/math/Vec3Conversions.cuh"
#include "AABB.cuh"
#include "AABB.hpp"

// Convert CPU AABB to CUDA AABB.
__host__ __device__ inline CudaAABB cpu_to_cuda_aabb(const AABB &cpu_aabb) {
  return CudaAABB(cpu_to_cuda_interval(cpu_aabb.x()),
                  cpu_to_cuda_interval(cpu_aabb.y()),
                  cpu_to_cuda_interval(cpu_aabb.z()));
}

// Convert CUDA AABB to CPU AABB.
__host__ __device__ inline AABB cuda_to_cpu_aabb(const CudaAABB &cuda_aabb) {
  return AABB(cuda_to_cpu_interval(cuda_aabb.x),
              cuda_to_cpu_interval(cuda_aabb.y),
              cuda_to_cpu_interval(cuda_aabb.z));
}

// Batch conversion functions for performance
__host__ void batch_cpu_to_cuda_aabb(const AABB *cpu_aabbs,
                                     CudaAABB *cuda_aabbs, int count);
__host__ void batch_cuda_to_cpu_aabb(const CudaAABB *cuda_aabbs,
                                     AABB *cpu_aabbs, int count);

#endif // USE_CUDA