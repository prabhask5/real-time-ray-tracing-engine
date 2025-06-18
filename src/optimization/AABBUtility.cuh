#pragma once

#ifdef USE_CUDA

#include "../core/Vec3Types.hpp"
#include "CudaAABB.cuh"
#include "CudaIntervalUtility.cuh"

// Adds an offset vector to an AABB — shifts the entire box by that vector.
__device__ __forceinline__ CudaAABB operator+(const CudaAABB &box,
                                              const CudaVec3 &offset) {
  return cuda_make_aabb(cuda_interval_offset(box.x, offset.x),
                        cuda_interval_offset(box.y, offset.y),
                        cuda_interval_offset(box.z, offset.z));
}

__device__ __forceinline__ CudaAABB operator+(const CudaVec3 &offset,
                                              const CudaAABB &box) {
  return box + offset;
}

// Bounding box comparison function — compares minimums of bounding box
// intervals along an axis.
__device__ __forceinline__ bool
cuda_bbox_compare_min(const CudaAABB &a, const CudaAABB &b, int axis_index) {
  const CudaInterval &a_axis = (axis_index == 0)   ? a.x
                               : (axis_index == 1) ? a.y
                                                   : a.z;
  const CudaInterval &b_axis = (axis_index == 0)   ? b.x
                               : (axis_index == 1) ? b.y
                                                   : b.z;
  return a_axis.min < b_axis.min;
}

__device__ __forceinline__ bool cuda_bbox_x_compare(const CudaAABB &a,
                                                    const CudaAABB &b) {
  return cuda_bbox_compare_min(a, b, 0);
}

__device__ __forceinline__ bool cuda_bbox_y_compare(const CudaAABB &a,
                                                    const CudaAABB &b) {
  return cuda_bbox_compare_min(a, b, 1);
}

__device__ __forceinline__ bool cuda_bbox_z_compare(const CudaAABB &a,
                                                    const CudaAABB &b) {
  return cuda_bbox_compare_min(a, b, 2);
}

#endif // USE_CUDA
