#pragma once

#ifdef USE_CUDA

#include "../core/Hittable.cuh"
#include "../core/HittableTypes.hpp"
#include "AABBConversions.cuh"
#include "BVHNode.cuh"

// Forward declarations.
struct CudaHittable;
CudaHittable cpu_to_cuda_hittable(const Hittable &cpu_hittable);

// Convert CPU BVHNode to CUDA BVHNode POD struct.
inline CudaBVHNode cpu_to_cuda_bvh_node(const BVHNode &cpu_bvh_node) {
  CudaHittable *left = new CudaHittable();
  CudaHittable *right = new CudaHittable();

  *left = cpu_to_cuda_hittable(*cpu_bvh_node.get_left());
  *right = cpu_to_cuda_hittable(*cpu_bvh_node.get_right());

  bool is_leaf = left->type != CudaHittableType::HITTABLE_BVH_NODE &&
                 right->type != CudaHittableType::HITTABLE_BVH_NODE;

  return cuda_make_bvh_node(left, right, is_leaf);
}

#endif // USE_CUDA