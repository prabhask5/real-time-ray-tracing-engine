#pragma once

#ifdef USE_CUDA

#include "../core/HitRecord.cuh"
#include "../core/Ray.cuh"
#include "../utils/math/Interval.cuh"
#include "../utils/math/Vec3Utility.cuh"
#include "AABB.cuh"
#include <curand_kernel.h>

// Forward declaration.
struct CudaHittable;

// POD struct representing a BVH node.
struct CudaBVHNode {
  CudaHittable *left;  // Left child hittable.
  CudaHittable *right; // Right child hittable.
  CudaAABB bbox;
  bool is_leaf; // True if this node contains actual objects, false if it has
                // child nodes.
};

// BVHNode initialization functions.

__host__ __device__ inline CudaBVHNode
cuda_make_bvh_node(CudaHittable *left, CudaHittable *right, bool is_leaf,
                   const CudaAABB &bbox) {
  CudaBVHNode node;
  node.left = left;
  node.right = right;
  node.is_leaf = is_leaf;
  node.bbox = bbox;
  return node;
}

// BVHNode utility functions.
__device__ bool cuda_bvh_node_hit(const CudaBVHNode &node, const CudaRay &ray,
                                  CudaInterval t_values, CudaHitRecord &record,
                                  curandState *rand_state);

__device__ double cuda_bvh_node_pdf_value(const CudaBVHNode &node,
                                          const CudaPoint3 &origin,
                                          const CudaVec3 &direction);

__device__ CudaVec3 cuda_bvh_node_random(const CudaBVHNode &node,
                                         const CudaPoint3 &origin,
                                         curandState *state);

__host__ __device__ inline CudaAABB
cuda_bvh_node_get_bounding_box(const CudaBVHNode &node) {
  return node.bbox;
}

// JSON serialization function for CudaBVHNode.
std::string cuda_json_bvh_node(const CudaBVHNode &obj);

#endif // USE_CUDA