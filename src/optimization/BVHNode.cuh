#pragma once

#ifdef USE_CUDA

#include "../core/HitRecord.cuh"
#include "../core/Hittable.cuh"
#include "../core/Ray.cuh"
#include "../utils/math/Interval.cuh"
#include "../utils/math/Vec3Utility.cuh"
#include "AABB.cuh"
#include <curand_kernel.h>

struct CudaBVHNode {
  CudaHittable left;  // Left child hittable.
  CudaHittable right; // Right child hittable.
  CudaAABB bbox;
  bool is_leaf; // True if this node contains actual objects, false if it has
                // child nodes.

  __device__ CudaBVHNode(const CudaHittable &_left, const CudaHittable &_right,
                         bool _is_leaf = false)
      : left(_left), right(_right), is_leaf(_is_leaf) {
    // Compute bounding box from children.
    CudaAABB left_box = left.get_bounding_box();
    CudaAABB right_box = right.get_bounding_box();
    bbox = CudaAABB(left_box, right_box);
  }

  // Hit test for BVH node.
  __device__ inline bool hit(const CudaRay &ray, CudaInterval t_values,
                             CudaHitRecord &record) {
    // Early exit if ray doesn't hit bounding box.
    if (!bbox.hit(ray, t_values))
      return false;

    CudaHitRecord temp_record;
    bool hit_left = left.hit(ray, t_values, temp_record);

    if (hit_left) {
      t_values.max = temp_record.t;
      record = temp_record;
    }

    bool hit_right = right.hit(ray, t_values, temp_record);
    if (hit_right)
      record = temp_record;

    return hit_left || hit_right;
  }

  // PDF value for BVH node (average of children).
  __device__ inline double pdf_value(const CudaPoint3 &origin,
                                     const CudaVec3 &direction) {
    return 0.5 * left.pdf_value(origin, direction) +
           0.5 * right.pdf_value(origin, direction);
  }

  // Random direction toward BVH node (random choice of children).
  __device__ inline CudaVec3 random(const CudaPoint3 &origin,
                                    curandState *state) {
    if (curand_uniform_double(state) < 0.5)
      return left.random(origin, state);
    return right.random(origin, state);
  }

  // Get bounding box for BVH node.
  __device__ inline CudaAABB get_bounding_box() { return bbox; }
};

#endif // USE_CUDA