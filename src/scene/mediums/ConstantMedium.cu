#ifdef USE_CUDA

#include "../../core/Hittable.cuh"
#include "../../utils/memory/CudaMemoryUtility.cuh"
#include "ConstantMedium.cuh"
#include <iomanip>
#include <sstream>

__device__ bool cuda_constant_medium_hit(const CudaConstantMedium &medium,
                                         const CudaRay &ray,
                                         CudaInterval t_range,
                                         CudaHitRecord &rec,
                                         curandState *rand_state) {
  CudaHitRecord rec1, rec2;

  if (!cuda_hittable_hit(*medium.boundary, ray, CUDA_UNIVERSE_INTERVAL, rec1,
                         rand_state))
    return false;

  if (!cuda_hittable_hit(*medium.boundary, ray,
                         cuda_make_interval(rec1.t + 0.0001, CUDA_INF), rec2,
                         rand_state))
    return false;

  if (rec1.t < t_range.min)
    rec1.t = t_range.min;
  if (rec2.t > t_range.max)
    rec2.t = t_range.max;

  if (rec1.t >= rec2.t)
    return false;

  if (rec1.t < 0)
    rec1.t = 0;

  double ray_length = cuda_vec3_length(ray.direction);
  double distance_inside = (rec2.t - rec1.t) * ray_length;

  double hit_distance =
      medium.neg_inv_density * log(cuda_random_double(rand_state));

  if (hit_distance > distance_inside)
    return false;

  rec.t = rec1.t + hit_distance / ray_length;
  rec.point = cuda_ray_at(ray, rec.t);
  rec.normal = cuda_make_vec3(1.0, 0.0, 0.0);
  rec.front_face = true;
  rec.material_index = medium.phase_function_index;

  return true;
}

__device__ CudaAABB
cuda_constant_medium_get_bounding_box(const CudaConstantMedium &medium) {
  return cuda_hittable_get_bounding_box(*medium.boundary);
}

// JSON serialization function for CudaConstantMedium.
std::string cuda_json_constant_medium(const CudaConstantMedium &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaConstantMedium\",";
  oss << "\"address\":\"" << &obj << "\",";
  if (obj.boundary) {
    CudaHittable host_boundary;
    cudaMemcpyDeviceToHostSafe(&host_boundary, obj.boundary, 1);
    oss << "\"boundary\":" << cuda_json_hittable(host_boundary) << ",";
  } else {
    oss << "\"boundary\":null,";
  }
  oss << "\"neg_inv_density\":" << obj.neg_inv_density << ",";
  oss << "\"phase_function_index\":" << obj.phase_function_index;
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA