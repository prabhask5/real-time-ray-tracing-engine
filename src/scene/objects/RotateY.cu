#ifdef USE_CUDA

#include "../../core/Hittable.cuh"
#include "../../utils/memory/CudaMemoryUtility.cuh"
#include "RotateY.cuh"
#include <iomanip>
#include <sstream>

__device__ bool cuda_rotate_y_hit(const CudaRotateY &rotate, const CudaRay &ray,
                                  CudaInterval t_values, CudaHitRecord &record,
                                  curandState *rand_state) {
  CudaPoint3 origin = cuda_make_vec3(
      rotate.cos_theta * ray.origin.x - rotate.sin_theta * ray.origin.z,
      ray.origin.y,
      rotate.sin_theta * ray.origin.x + rotate.cos_theta * ray.origin.z);

  CudaVec3 direction = cuda_make_vec3(
      rotate.cos_theta * ray.direction.x - rotate.sin_theta * ray.direction.z,
      ray.direction.y,
      rotate.sin_theta * ray.direction.x + rotate.cos_theta * ray.direction.z);

  CudaRay rotated_ray = cuda_make_ray(origin, direction, ray.time);

  if (!cuda_hittable_hit(*rotate.object, rotated_ray, t_values, record,
                         rand_state))
    return false;

  record.point = cuda_make_vec3(
      rotate.cos_theta * record.point.x + rotate.sin_theta * record.point.z,
      record.point.y,
      -rotate.sin_theta * record.point.x + rotate.cos_theta * record.point.z);

  record.normal = cuda_make_vec3(
      rotate.cos_theta * record.normal.x + rotate.sin_theta * record.normal.z,
      record.normal.y,
      -rotate.sin_theta * record.normal.x + rotate.cos_theta * record.normal.z);

  return true;
}

__device__ double cuda_rotate_y_pdf_value(const CudaRotateY &rotate,
                                          const CudaPoint3 &origin,
                                          const CudaVec3 &direction) {
  CudaPoint3 rotated_origin = cuda_make_vec3(
      rotate.cos_theta * origin.x - rotate.sin_theta * origin.z, origin.y,
      rotate.sin_theta * origin.x + rotate.cos_theta * origin.z);

  CudaVec3 rotated_direction = cuda_make_vec3(
      rotate.cos_theta * direction.x - rotate.sin_theta * direction.z,
      direction.y,
      rotate.sin_theta * direction.x + rotate.cos_theta * direction.z);

  return cuda_hittable_pdf_value(*rotate.object, rotated_origin,
                                 rotated_direction);
}

__device__ CudaVec3 cuda_rotate_y_random(const CudaRotateY &rotate,
                                         const CudaPoint3 &origin,
                                         curandState *state) {
  CudaPoint3 rotated_origin = cuda_make_vec3(
      rotate.cos_theta * origin.x - rotate.sin_theta * origin.z, origin.y,
      rotate.sin_theta * origin.x + rotate.cos_theta * origin.z);

  CudaVec3 obj_dir =
      cuda_hittable_random(*rotate.object, rotated_origin, state);

  return cuda_make_vec3(
      rotate.cos_theta * obj_dir.x + rotate.sin_theta * obj_dir.z, obj_dir.y,
      -rotate.sin_theta * obj_dir.x + rotate.cos_theta * obj_dir.z);
}

// JSON serialization function for CudaRotateY.
std::string cuda_json_rotate_y(const CudaRotateY &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaRotateY\",";
  oss << "\"address\":\"" << &obj << "\",";
  if (obj.object) {
    CudaHittable host_object;
    cudaMemcpyDeviceToHostSafe(&host_object, obj.object, 1);
    oss << "\"object\":" << cuda_json_hittable(host_object) << ",";
  } else {
    oss << "\"object\":null,";
  }
  oss << "\"sin_theta\":" << obj.sin_theta << ",";
  oss << "\"cos_theta\":" << obj.cos_theta << ",";
  oss << "\"bbox\":" << cuda_json_aabb(obj.bbox);
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA