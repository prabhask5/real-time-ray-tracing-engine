#pragma once

#ifdef USE_CUDA

#include "CudaAABB.cuh"
#include "CudaHitRecord.cuh"
#include "CudaInterval.cuh"
#include "CudaRay.cuh"
#include "CudaVec3Utility.cuh"
#include <curand_kernel.h>

struct CudaBVHNode {
  void *left;
  void *right;
  CudaAABB bbox;
};

// Manually-dispatched hit test
__device__ __forceinline__ bool cuda_bvhnode_hit(
    const CudaBVHNode *node, const CudaRay &ray, CudaInterval t_values,
    CudaHitRecord &record,
    bool (*hit_fn)(void *, const CudaRay &, CudaInterval, CudaHitRecord &)) {

  if (!cuda_aabb_hit(node->bbox, ray, t_values))
    return false;

  CudaHitRecord temp_record;
  bool hit_left = hit_fn(node->left, ray, t_values, temp_record);

  if (hit_left) {
    t_values.max_val = temp_record.t;
    record = temp_record;
  }

  bool hit_right = hit_fn(node->right, ray, t_values, temp_record);
  if (hit_right)
    record = temp_record;

  return hit_left || hit_right;
}

// Returns the PDF value from a BVH node
__device__ __forceinline__ double cuda_bvhnode_pdf_value(
    const CudaBVHNode *node, const CudaVec3 &origin, const CudaVec3 &direction,
    double (*pdf_fn)(void *, const CudaVec3 &, const CudaVec3 &)) {

  return 0.5 * pdf_fn(node->left, origin, direction) +
         0.5 * pdf_fn(node->right, origin, direction);
}

// Generates a random direction toward a BVH subtree
__device__ __forceinline__ CudaVec3 cuda_bvhnode_random(
    const CudaBVHNode *node, const CudaVec3 &origin, curandState *rand_state,
    CudaVec3 (*rand_fn)(void *, const CudaVec3 &, curandState *)) {

  if (curand(rand_state) & 1)
    return rand_fn(node->left, origin, rand_state);
  return rand_fn(node->right, origin, rand_state);
}

#endif // USE_CUDA
