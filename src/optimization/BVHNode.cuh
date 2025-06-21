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

// Represents a bounding box that can contain multiple inner bounding boxes as
// children. A leaf node contains 1 or a few geometric objects. Optimizes the
// ray hit algorithm by ignoring all the inner bounding boxes in which the ray
// doesn't interact with the enclosing bounding box.
struct CudaBVHNode {
  CudaHittable *left;  // Left child hittable.
  CudaHittable *right; // Right child hittable.
  CudaAABB bbox;
  bool is_leaf; // True if this node contains actual objects, false if it has
                // child nodes.

  __device__ CudaBVHNode() {} // Default constructor.

  __device__ CudaBVHNode(CudaHittable *_left, CudaHittable *_right,
                         bool _is_leaf = false);

  __host__ __device__ CudaBVHNode(CudaHittable *_left, CudaHittable *_right,
                                  bool _is_leaf, const CudaAABB &_bbox)
      : left(_left), right(_right), is_leaf(_is_leaf), bbox(_bbox) {}

  // Hit test for BVH node.
  __device__ bool hit(const CudaRay &ray, CudaInterval t_values,
                      CudaHitRecord &record, curandState *rand_state) const;

  // PDF value for BVH node (average of children).
  __device__ double pdf_value(const CudaPoint3 &origin,
                              const CudaVec3 &direction) const;

  // Random direction toward BVH node (random choice of children).
  __device__ CudaVec3 random(const CudaPoint3 &origin,
                             curandState *state) const;

  // Get bounding box for BVH node.
  __device__ inline CudaAABB get_bounding_box() const { return bbox; }
};

#endif // USE_CUDA