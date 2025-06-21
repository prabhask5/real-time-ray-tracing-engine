#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../optimization/AABB.cuh"
#include "../../optimization/AABBUtility.cuh"
#include "../../utils/math/Interval.cuh"

// Wrapper class. Moves (translates) a hittable object by an offset vector in
// world space.
struct CudaTranslate {
  const CudaHittable *object; // Can be any hittable (e.g. CudaSphere, CudaBox).
  CudaVec3 offset;
  CudaAABB bbox;

  __device__ CudaTranslate() {} // Default constructor.

  // Initializes the translated object wrapper.
  __device__ CudaTranslate(const CudaHittable *_object,
                           const CudaVec3 &_offset);

  // Initialize translate from direct members.
  __host__ __device__ CudaTranslate(const CudaHittable *_object,
                                    const CudaVec3 &_offset,
                                    const CudaAABB &_bbox)
      : object(_object), offset(_offset), bbox(_bbox) {}

  // Hit test for a translated object.
  __device__ bool hit(const CudaRay &ray, CudaInterval t_range,
                      CudaHitRecord &rec, curandState *rand_state) const;

  // PDF value for the translated object.
  __device__ double pdf_value(const CudaPoint3 &origin,
                              const CudaVec3 &direction) const;

  // Random direction toward the translated object.
  __device__ CudaVec3 random(const CudaPoint3 &origin,
                             curandState *state) const;

  // Get bounding box for the translated object.
  __device__ inline CudaAABB get_bounding_box() const { return bbox; }
};

#endif // USE_CUDA
