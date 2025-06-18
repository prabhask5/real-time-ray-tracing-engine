#pragma once

#ifdef USE_CUDA

#include "../../core/AABB.cuh"
#include "../../core/HitRecord.cuh"
#include "../../core/Interval.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.hpp"
#include "CudaHittable.cuh"
#include <curand_kernel.h>

// Represents a hittable object rotated around the Y-axis by a fixed angle.
struct CudaRotateY {
  const CudaHittable *object;
  double sin_theta;
  double cos_theta;
  CudaAABB bbox;
};

// Initializes a CudaRotateY wrapper given an object and angle in degrees.
__host__ __device__ inline void
cuda_init_rotate_y(CudaRotateY &rot, const CudaHittable *object,
                   double angle_degrees, const CudaAABB &original_bbox) {

  double radians = angle_degrees * 0.017453292519943295; // PI / 180
  rot.object = object;
  rot.sin_theta = sin(radians);
  rot.cos_theta = cos(radians);

  CudaPoint3 min(CUDA_INF, CUDA_INF, CUDA_INF);
  CudaPoint3 max(-CUDA_INF, -CUDA_INF, -CUDA_INF);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        double x = i ? original_bbox.x().max() : original_bbox.x().min();
        double y = j ? original_bbox.y().max() : original_bbox.y().min();
        double z = k ? original_bbox.z().max() : original_bbox.z().min();

        double new_x = rot.cos_theta * x + rot.sin_theta * z;
        double new_z = -rot.sin_theta * x + rot.cos_theta * z;

        CudaVec3 tester(new_x, y, new_z);
        for (int c = 0; c < 3; c++) {
          min[c] = fmin(min[c], tester[c]);
          max[c] = fmax(max[c], tester[c]);
        }
      }
    }
  }

  rot.bbox = CudaAABB(min, max);
}

// Returns the rotated AABB
__device__ inline CudaAABB cuda_rotate_y_bbox(const CudaRotateY &rot) {
  return rot.bbox;
}

// Tests hit against the rotated object
__device__ inline bool cuda_rotate_y_hit(const CudaRotateY &rot,
                                         const CudaRay &ray,
                                         CudaInterval t_values,
                                         CudaHitRecord &record,
                                         curandState *rand_state) {

  // Transform ray into object space
  CudaPoint3 origin(
      rot.cos_theta * ray.origin.x() - rot.sin_theta * ray.origin.z(),
      ray.origin.y(),
      rot.sin_theta * ray.origin.x() + rot.cos_theta * ray.origin.z());

  CudaVec3 direction(
      rot.cos_theta * ray.direction.x() - rot.sin_theta * ray.direction.z(),
      ray.direction.y(),
      rot.sin_theta * ray.direction.x() + rot.cos_theta * ray.direction.z());

  CudaRay rotated_ray(origin, direction, ray.time);

  if (!cuda_hittable_hit(*rot.object, rotated_ray, t_values, record,
                         rand_state))
    return false;

  // Transform point and normal back to world space
  CudaPoint3 p(
      rot.cos_theta * record.point.x() + rot.sin_theta * record.point.z(),
      record.point.y(),
      -rot.sin_theta * record.point.x() + rot.cos_theta * record.point.z());

  CudaVec3 n(
      rot.cos_theta * record.normal.x() + rot.sin_theta * record.normal.z(),
      record.normal.y(),
      -rot.sin_theta * record.normal.x() + rot.cos_theta * record.normal.z());

  record.point = p;
  record.normal = n;
  record.front_face =
      cuda_dot_product(rotated_ray.direction, record.normal) < 0;

  return true;
}

#endif // USE_CUDA
