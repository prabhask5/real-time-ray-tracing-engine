#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Hittable.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../optimization/AABB.cuh"
#include "../../utils/math/Interval.cuh"
#include "../../utils/math/Utility.cuh"
#include "../../utils/math/Vec3Utility.cuh"
#include <curand_kernel.h>

// Represents a hittable object rotated around the Y-axis by a fixed angle.
struct CudaRotateY {
  const CudaHittable *object;
  double sin_theta;
  double cos_theta;
  CudaAABB bbox;

  // Initializes a CudaRotateY wrapper given an object and angle in degrees.
  __device__ CudaRotateY(const CudaHittable *_object, double angle_degrees,
                         const CudaAABB &original_bbox)
      : object(_object) {

    double radians = cuda_degrees_to_radians(angle_degrees);
    sin_theta = sin(radians);
    cos_theta = cos(radians);

    CudaPoint3 min(CUDA_INF, CUDA_INF, CUDA_INF);
    CudaPoint3 max(-CUDA_INF, -CUDA_INF, -CUDA_INF);

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          double x = i ? original_bbox.x.max : original_bbox.x.min;
          double y = j ? original_bbox.y.max : original_bbox.y.min;
          double z = k ? original_bbox.z.max : original_bbox.z.min;

          double new_x = cos_theta * x + sin_theta * z;
          double new_z = -sin_theta * x + cos_theta * z;

          CudaVec3 tester(new_x, y, new_z);
          for (int c = 0; c < 3; c++) {
            min[c] = fmin(min[c], tester[c]);
            max[c] = fmax(max[c], tester[c]);
          }
        }
      }
    }

    bbox = CudaAABB(min, max);
  }

  // Tests hit against the rotated object.
  __device__ inline bool hit(const CudaRay &ray, CudaInterval t_values,
                             CudaHitRecord &record, curandState *rand_state) {

    // Transform ray into object space.
    CudaPoint3 origin(cos_theta * ray.origin.x - sin_theta * ray.origin.z,
                      ray.origin.y,
                      sin_theta * ray.origin.x + cos_theta * ray.origin.z);

    CudaVec3 direction(
        cos_theta * ray.direction.x - sin_theta * ray.direction.z,
        ray.direction.y,
        sin_theta * ray.direction.x + cos_theta * ray.direction.z);

    CudaRay rotated_ray = CudaRay(origin, direction, ray.time);

    if (!object->hit(rotated_ray, t_values, record, rand_state))
      return false;

    // Transform point and normal back to world space.
    CudaPoint3 p(cos_theta * record.point.x + sin_theta * record.point.z,
                 record.point.y,
                 -sin_theta * record.point.x + cos_theta * record.point.z);

    CudaVec3 n(cos_theta * record.normal.x + sin_theta * record.normal.z,
               record.normal.y,
               -sin_theta * record.normal.x + cos_theta * record.normal.z);

    record.point = p;
    cuda_set_face_normal(record, ray, n);

    return true;
  }

  // PDF value for the rotated object.
  __device__ inline double pdf_value(const CudaPoint3 &origin,
                                     const CudaVec3 &direction) {
    // Transform origin and direction to object space.
    CudaPoint3 rotated_origin(cos_theta * origin.x - sin_theta * origin.z,
                              origin.y,
                              sin_theta * origin.x + cos_theta * origin.z);

    CudaVec3 rotated_direction(
        cos_theta * direction.x - sin_theta * direction.z, direction.y,
        sin_theta * direction.x + cos_theta * direction.z);

    return object->pdf_value(rotated_origin, rotated_direction);
  }

  // Random direction toward the rotated object
  __device__ inline CudaVec3 random(const CudaPoint3 &origin,
                                    curandState *state) {
    // Transform origin to object space.
    CudaPoint3 rotated_origin(cos_theta * origin.x - sin_theta * origin.z,
                              origin.y,
                              sin_theta * origin.x + cos_theta * origin.z);

    // Get random direction in object space.
    CudaVec3 obj_dir = object->random(rotated_origin, state);

    // Transform back to world space.
    return CudaVec3(cos_theta * obj_dir.x + sin_theta * obj_dir.z, obj_dir.y,
                    -sin_theta * obj_dir.x + cos_theta * obj_dir.z);
  }

  // Get bounding box for the rotated object.
  __device__ inline CudaAABB get_bounding_box() { return bbox; }
};

#endif // USE_CUDA
