#pragma once

#ifdef USE_CUDA

#include "AABB.cuh"
#include "HitRecord.cuh"
#include "Interval.cuh"
#include "Ray.cuh"
#include "Vec3Types.cuh"

// Enumeration for CUDA Hittable object types (used for manual dispatch).
enum class CudaHittableType {
  NONE = 0,
  SPHERE = 1,
  PLANE = 2,
  BVH_NODE = 3,
  CONSTANT_MEDIUM = 4,
  ROTATE_Y = 5,
  TRANSLATE = 6,
  LIST = 7,
};

// A CUDA-compatible Hittable interface implemented using manual dispatch.
// Each hittable object must implement these function signatures.
using CudaHitFn = bool (*)(const void *hittable_data, const CudaRay &,
                           CudaInterval, CudaHitRecord &);
using CudaPDFValueFn = double (*)(const void *hittable_data, const CudaPoint3 &,
                                  const CudaVec3 &);
using CudaRandomFn = CudaVec3 (*)(const void *hittable_data,
                                  const CudaPoint3 &);
using CudaBBoxFn = CudaAABB (*)(const void *hittable_data);

// Dispatcher for Hittable interfaces — must be filled per object.
struct CudaHittableVTable {
  CudaHitFn hit;
  CudaPDFValueFn pdf_value;
  CudaRandomFn random;
  CudaBBoxFn bounding_box;
};

// Unified CUDA Hittable wrapper.
// This allows all types of hittables to be passed around uniformly.
struct CudaHittable {
  void *data;
  CudaHittableVTable *vtable;
};

// Invokes hit(...) on a generic CUDA hittable.
__device__ inline bool cuda_hittable_hit(const CudaHittable &h,
                                         const CudaRay &r, CudaInterval t_range,
                                         CudaHitRecord &rec) {
  return h.vtable->hit(h.data, r, t_range, rec);
}

// Invokes pdf_value(...) on a generic CUDA hittable.
__device__ inline double cuda_hittable_pdf_value(const CudaHittable &h,
                                                 const CudaPoint3 &origin,
                                                 const CudaVec3 &direction) {
  return h.vtable->pdf_value(h.data, origin, direction);
}

// Invokes random(...) on a generic CUDA hittable.
__device__ inline CudaVec3 cuda_hittable_random(const CudaHittable &h,
                                                const CudaPoint3 &origin) {
  return h.vtable->random(h.data, origin);
}

// Invokes get_bounding_box(...) on a generic CUDA hittable.
__device__ inline CudaAABB cuda_hittable_bounding_box(const CudaHittable &h) {
  return h.vtable->bounding_box(h.data);
}

#endif // USE_CUDA
