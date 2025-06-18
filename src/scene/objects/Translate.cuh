#pragma once

#ifdef USE_CUDA

#include "../../core/AABB.cuh"
#include "../../core/HitRecord.cuh"
#include "../../core/HittableUnion.cuh" // Manual dispatch of inner hittable
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.hpp"

// CUDA-compatible translation wrapper for a hittable object.
struct CudaTranslate {
  CudaHittableUnion object; // Can be any hittable (e.g. CudaSphere, CudaBox)
  CudaVec3 offset;
  CudaAABB bbox;
};

// Initializes the translated object wrapper.
__host__ __device__ inline void
cuda_init_translate(CudaTranslate &t, const CudaHittableUnion &object,
                    const CudaVec3 &offset) {
  t.object = object;
  t.offset = offset;
  t.bbox = cuda_aabb_offset(cuda_get_bbox(object), offset);
}

// Returns the bounding box of the translated object.
__device__ inline CudaAABB cuda_translate_bbox(const CudaTranslate &t) {
  return t.bbox;
}

// Hit test for a translated object.
__device__ inline bool cuda_translate_hit(const CudaTranslate &t,
                                          const CudaRay &ray,
                                          CudaInterval t_range,
                                          CudaHitRecord &rec) {

  // Move the ray backwards by the translation offset.
  CudaRay moved_ray(ray.origin - t.offset, ray.direction, ray.time);

  // Dispatch the hit function of the wrapped object.
  if (!cuda_hittable_hit(t.object, moved_ray, t_range, rec))
    return false;

  // Offset the hit point back to world space.
  rec.point += t.offset;

  return true;
}

#endif // USE_CUDA
