#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../optimization/AABB.cuh"
#include "../../optimization/AABBUtility.cuh"
#include "../../utils/math/Interval.cuh"
#include <curand_kernel.h>

// POD struct representing a translated hittable object.
struct CudaTranslate {
  const CudaHittable *object; // Can be any hittable (e.g. CudaSphere, CudaBox).
  CudaVec3 offset;
  CudaAABB bbox;
};

// Translate initialization functions.
__device__ inline CudaTranslate cuda_make_translate(const CudaHittable *object,
                                                    const CudaVec3 &offset);

__host__ __device__ inline CudaTranslate
cuda_make_translate(const CudaHittable *object, const CudaVec3 &offset,
                    const CudaAABB &bbox) {
  CudaTranslate translate;
  translate.object = object;
  translate.offset = offset;
  translate.bbox = bbox;
  return translate;
}

// Translate utility functions.
__device__ bool cuda_translate_hit(const CudaTranslate &translate,
                                   const CudaRay &ray, CudaInterval t_range,
                                   CudaHitRecord &rec, curandState *rand_state);

__device__ double cuda_translate_pdf_value(const CudaTranslate &translate,
                                           const CudaPoint3 &origin,
                                           const CudaVec3 &direction);

__device__ CudaVec3 cuda_translate_random(const CudaTranslate &translate,
                                          const CudaPoint3 &origin,
                                          curandState *state);

__device__ inline CudaAABB
cuda_translate_get_bounding_box(const CudaTranslate &translate) {
  return translate.bbox;
}

#endif // USE_CUDA
