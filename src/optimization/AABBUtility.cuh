#pragma once

#ifdef USE_CUDA

#include "../core/Vec3Types.cuh"
#include "../utils/math/IntervalUtility.cuh"
#include "AABB.cuh"

// Adds an offset vector to an AABB — shifts the entire box by that vector.
__device__ __forceinline__ CudaAABB operator+(const CudaAABB &box,
                                              const CudaVec3 &offset) {
  return cuda_make_aabb(box.x + offset.x, box.y + offset.y, box.z + offset.z);
}

__device__ __forceinline__ CudaAABB operator+(const CudaVec3 &offset,
                                              const CudaAABB &box) {
  return box + offset;
}

// Bounding box comparison function — compares minimums of bounding box
// intervals along an axis.
__device__ __forceinline__ bool
cuda_bbox_compare(const CudaAABB &a, const CudaAABB &b, int axis_index) {
  const CudaInterval &a_axis = cuda_aabb_get_axis_interval(a, axis_index);
  const CudaInterval &b_axis = cuda_aabb_get_axis_interval(b, axis_index);
  return a_axis.min < b_axis.min;
}

__device__ __forceinline__ bool cuda_bbox_x_compare(const CudaAABB &a,
                                                    const CudaAABB &b) {
  return cuda_bbox_compare(a, b, 0);
}

__device__ __forceinline__ bool cuda_bbox_y_compare(const CudaAABB &a,
                                                    const CudaAABB &b) {
  return cuda_bbox_compare(a, b, 1);
}

__device__ __forceinline__ bool cuda_bbox_z_compare(const CudaAABB &a,
                                                    const CudaAABB &b) {
  return cuda_bbox_compare(a, b, 2);
}

#endif // USE_CUDA
