#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Hittable.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../optimization/AABB.cuh"
#include "../../optimization/AABBUtility.cuh"
#include "../../utils/math/Interval.cuh"

// CUDA-compatible translation wrapper for a hittable object.
struct CudaTranslate {
  const CudaHittable *object; // Can be any hittable (e.g. CudaSphere, CudaBox)
  CudaVec3 offset;
  CudaAABB bbox;

  // Initializes the translated object wrapper.
  __device__ CudaTranslate(const CudaHittable *_object, const CudaVec3 &_offset)
      : object(_object), offset(_offset) {
    bbox = object->get_bounding_box() + offset;
  }

  // Hit test for a translated object.
  __device__ inline bool hit(const CudaRay &ray, CudaInterval t_range,
                             CudaHitRecord &rec, curandState *rand_state) {

    // Move the ray backwards by the translation offset.
    CudaRay moved_ray = CudaRay(ray.origin - offset, ray.direction, ray.time);

    // Dispatch the hit function of the wrapped object.
    if (!object->hit(moved_ray, t_range, rec, rand_state))
      return false;

    // Offset the hit point back to world space.
    rec.point += offset;

    return true;
  }

  // PDF value for the translated object.
  __device__ inline double pdf_value(const CudaPoint3 &origin,
                                     const CudaVec3 &direction) {
    // Move the origin backwards by the translation offset.
    return object->pdf_value(origin - offset, direction);
  }

  // Random direction toward the translated object.
  __device__ inline CudaVec3 random(const CudaPoint3 &origin,
                                    curandState *state) {
    // Move the origin backwards by the translation offset.
    return object->random(origin - offset, state);
  }

  // Get bounding box for the translated object.
  __device__ inline CudaAABB get_bounding_box() { return bbox; }
};

#endif // USE_CUDA
