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

// Wrapper class. Rotates a hittable object around the Y-axis by a given angle.
struct CudaRotateY {
  const CudaHittable *object;
  double sin_theta;
  double cos_theta;
  CudaAABB bbox;

  __device__ CudaRotateY() {} // Default constructor.

  // Initializes a CudaRotateY wrapper given an object and angle in degrees.
  __device__ CudaRotateY(const CudaHittable *_object, double angle_degrees);

  __host__ __device__ CudaRotateY(const CudaHittable *_object,
                                  double _sin_theta, double _cos_theta,
                                  const CudaAABB &_bbox)
      : object(_object), sin_theta(_sin_theta), cos_theta(_cos_theta),
        bbox(_bbox) {}

  // Tests hit against the rotated object.
  __device__ bool hit(const CudaRay &ray, CudaInterval t_values,
                      CudaHitRecord &record, curandState *rand_state) const;

  // PDF value for the rotated object.
  __device__ double pdf_value(const CudaPoint3 &origin,
                              const CudaVec3 &direction) const;

  // Random direction toward the rotated object.
  __device__ CudaVec3 random(const CudaPoint3 &origin,
                             curandState *state) const;

  // Get bounding box for the rotated object.
  __device__ inline CudaAABB get_bounding_box() const { return bbox; }
};

#endif // USE_CUDA
