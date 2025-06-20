#pragma once

#ifdef USE_CUDA

#include "../core/HittableTypes.hpp"
#include "AABBConversions.cuh"
#include "BVHNode.cuh"

// Forward declarations.
class BVHNode;
struct CudaHittable;
class Hittable;

// Forward declaration of conversion functions
inline CudaHittable cpu_to_cuda_hittable(const HittablePtr &cpu_hittable);
inline HittablePtr cuda_to_cpu_hittable(const CudaHittable &cuda_hittable);

// Convert CPU BVHNode to CUDA BVHNode.
inline CudaBVHNode cpu_to_cuda_bvh_node(const BVHNode &cpu_bvh_node) {
  // Extract properties from CPU BVHNode using reflection/getters.
  // Note: This assumes BVHNode has getter methods for left, right, and bounding
  // box.

  // Get left and right children.
  auto cpu_left = cpu_bvh_node.get_left();
  auto cpu_right = cpu_bvh_node.get_right();

  // Convert children to CUDA format.
  CudaHittable cuda_left = cpu_to_cuda_hittable(cpu_left);
  CudaHittable cuda_right = cpu_to_cuda_hittable(cpu_right);

  // Create CUDA BVH node.
  CudaBVHNode cuda_bvh_node(cuda_left, cuda_right, cpu_bvh_node.is_leaf());

  return cuda_bvh_node;
}

// Convert CUDA BVHNode to CPU BVHNode.
inline BVHNode cuda_to_cpu_bvh_node(const CudaBVHNode &cuda_bvh_node) {
  // Convert children back to CPU format.
  auto cpu_left = cuda_to_cpu_hittable(cuda_bvh_node.left);
  auto cpu_right = cuda_to_cpu_hittable(cuda_bvh_node.right);

  // Create CPU BVH node with converted children.
  return BVHNode(cpu_left, cpu_right);
}

#endif // USE_CUDA