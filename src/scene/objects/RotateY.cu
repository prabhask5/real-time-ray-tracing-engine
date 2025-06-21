#ifdef USE_CUDA

#include "../../core/Hittable.cuh"
#include "RotateY.cuh"

__device__ CudaRotateY::CudaRotateY(const CudaHittable *_object,
                                    double angle_degrees)
    : object(_object) {

  double radians = cuda_degrees_to_radians(angle_degrees);
  sin_theta = sin(radians);
  cos_theta = cos(radians);
  bbox = object->get_bounding_box();

  CudaPoint3 min(CUDA_INF, CUDA_INF, CUDA_INF);
  CudaPoint3 max(-CUDA_INF, -CUDA_INF, -CUDA_INF);

  // Computes the diagonal 3D space points of the rotated bbox by using
  // precomputed sin and cos theta values.
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        double x = i * bbox.x.max + (1 - i) * bbox.x.min;
        double y = j * bbox.y.max + (1 - j) * bbox.y.min;
        double z = k * bbox.z.max + (1 - k) * bbox.z.min;

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

__device__ bool CudaRotateY::hit(const CudaRay &ray, CudaInterval t_values,
                                 CudaHitRecord &record,
                                 curandState *rand_state) const {

  // Transform ray into object space.
  CudaPoint3 origin(cos_theta * ray.origin.x - sin_theta * ray.origin.z,
                    ray.origin.y,
                    sin_theta * ray.origin.x + cos_theta * ray.origin.z);

  CudaVec3 direction(cos_theta * ray.direction.x - sin_theta * ray.direction.z,
                     ray.direction.y,
                     sin_theta * ray.direction.x + cos_theta * ray.direction.z);

  CudaRay rotated_ray = CudaRay(origin, direction, ray.time);

  // Determine whether an intersection exists in object space (and if so,
  // where).

  if (!object->hit(rotated_ray, t_values, record, rand_state))
    return false;

  // Transform point and normal back to world space.
  record.point = CudaPoint3(
      cos_theta * record.point.x + sin_theta * record.point.z, record.point.y,
      -sin_theta * record.point.x + cos_theta * record.point.z);

  record.normal =
      CudaVec3(cos_theta * record.normal.x + sin_theta * record.normal.z,
               record.normal.y,
               -sin_theta * record.normal.x + cos_theta * record.normal.z);

  return true;
}

__device__ double CudaRotateY::pdf_value(const CudaPoint3 &origin,
                                         const CudaVec3 &direction) const {
  // Transform origin and direction to object space.
  CudaPoint3 rotated_origin(cos_theta * origin.x - sin_theta * origin.z,
                            origin.y,
                            sin_theta * origin.x + cos_theta * origin.z);

  CudaVec3 rotated_direction(cos_theta * direction.x - sin_theta * direction.z,
                             direction.y,
                             sin_theta * direction.x + cos_theta * direction.z);

  return object->pdf_value(rotated_origin, rotated_direction);
}

__device__ CudaVec3 CudaRotateY::random(const CudaPoint3 &origin,
                                        curandState *state) const {
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

#endif // USE_CUDA