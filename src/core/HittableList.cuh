#pragma once

#ifdef USE_CUDA

#include "../optimization/AABB.cuh"
#include "../utils/math/Interval.cuh"
#include "HitRecord.cuh"
#include "Ray.cuh"
#include "Vec3Types.cuh"
#include <curand_kernel.h>

// Forward declaration.
struct CudaHittable;

// Maximum objects per list.
static const int MAX_HITTABLES_PER_LIST = 1024;

// POD struct representing a list of hittables.
struct CudaHittableList {
  CudaHittable *hittables; // Pointer to array of hittables.
  int count;
  CudaAABB bbox;
};

// HittableList initialization functions.
__device__ CudaHittableList cuda_make_hittable_list(CudaHittable *hittables,
                                                    int count);

__host__ __device__ inline CudaHittableList
cuda_make_hittable_list(CudaHittable *hittables, int count,
                        const CudaAABB &bbox) {
  CudaHittableList list;
  list.hittables = hittables;
  list.count = count;
  list.bbox = bbox;
  return list;
}

// HittableList utility functions.
__device__ bool cuda_hittable_list_hit(const CudaHittableList &list,
                                       const CudaRay &ray, CudaInterval t_range,
                                       CudaHitRecord &out_rec,
                                       curandState *rand_state);

__device__ double cuda_hittable_list_pdf_value(const CudaHittableList &list,
                                               const CudaPoint3 &origin,
                                               const CudaVec3 &direction);

__device__ CudaVec3 cuda_hittable_list_random(const CudaHittableList &list,
                                              const CudaPoint3 &origin,
                                              curandState *state);

__device__ inline CudaAABB
cuda_hittable_list_get_bounding_box(const CudaHittableList &list) {
  return list.bbox;
}

#endif // USE_CUDA