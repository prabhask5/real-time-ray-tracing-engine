#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../optimization/AABB.cuh"
#include "../../utils/math/Interval.cuh"
#include "../../utils/math/Utility.cuh"
#include "../../utils/math/Vec3Utility.cuh"
#include <curand_kernel.h>

// POD struct that rotates a hittable object around the Y-axis.
struct CudaRotateY {
  const CudaHittable *object;
  double sin_theta;
  double cos_theta;
  CudaAABB bbox;
};

// RotateY initialization functions.

__host__ __device__ inline CudaRotateY
cuda_make_rotate_y(const CudaHittable *object, double sin_theta,
                   double cos_theta, const CudaAABB &bbox) {
  CudaRotateY rotate;
  rotate.object = object;
  rotate.sin_theta = sin_theta;
  rotate.cos_theta = cos_theta;
  rotate.bbox = bbox;
  return rotate;
}

// RotateY utility functions.
__device__ bool cuda_rotate_y_hit(const CudaRotateY &rotate, const CudaRay &ray,
                                  CudaInterval t_values, CudaHitRecord &record,
                                  curandState *rand_state);

__device__ double cuda_rotate_y_pdf_value(const CudaRotateY &rotate,
                                          const CudaPoint3 &origin,
                                          const CudaVec3 &direction);

__device__ CudaVec3 cuda_rotate_y_random(const CudaRotateY &rotate,
                                         const CudaPoint3 &origin,
                                         curandState *state);

__host__ __device__ inline CudaAABB
cuda_rotate_y_get_bounding_box(const CudaRotateY &rotate) {
  return rotate.bbox;
}

#endif // USE_CUDA
