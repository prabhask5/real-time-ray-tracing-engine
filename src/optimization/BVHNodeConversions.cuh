#pragma once

#ifdef USE_CUDA

// Forward declarations
struct CudaHittable;
CudaHittable cpu_to_cuda_hittable(const Hittable &cpu_hittable);
HittablePtr cuda_to_cpu_hittable(const CudaHittable &cuda_hittable);

#include "../core/Hittable.cuh"
#include "../core/HittableTypes.hpp"
#include "AABBConversions.cuh"
#include "BVHNode.cuh"

// Convert CPU BVHNode to CUDA BVHNode.
inline CudaBVHNode cpu_to_cuda_bvh_node(const BVHNode &cpu_bvh_node) {
  CudaHittable *left = new CudaHittable();
  CudaHittable *right = new CudaHittable();

  *left = cpu_to_cuda_hittable(*cpu_bvh_node.get_left());
  *right = cpu_to_cuda_hittable(*cpu_bvh_node.get_right());

  bool is_leaf = left->type != CudaHittableType::HITTABLE_BVH_NODE &&
                 right->type != CudaHittableType::HITTABLE_BVH_NODE;

  return CudaBVHNode(left, right, is_leaf);
}

// Convert CUDA BVHNode to CPU BVHNode.
inline BVHNode cuda_to_cpu_bvh_node(const CudaBVHNode &cuda_bvh_node) {
  // Convert children back to CPU format.
  HittablePtr cpu_left = nullptr;
  HittablePtr cpu_right = nullptr;

  if (cuda_bvh_node.left) {
    cpu_left = cuda_to_cpu_hittable(*cuda_bvh_node.left);
  }
  if (cuda_bvh_node.right) {
    cpu_right = cuda_to_cpu_hittable(*cuda_bvh_node.right);
  }

  // Create CPU BVH node with converted children.
  return BVHNode(cpu_left, cpu_right);
}

#endif // USE_CUDA