#ifdef USE_CUDA

#include "../../core/Hittable.cuh"
#include "../../utils/memory/CudaMemoryUtility.cuh"
#include "Translate.cuh"
#include <iomanip>
#include <sstream>

__device__ bool cuda_translate_hit(const CudaTranslate &translate,
                                   const CudaRay &ray, CudaInterval t_range,
                                   CudaHitRecord &rec,
                                   curandState *rand_state) {
  // Move the ray backwards by the translation offset.
  CudaRay moved_ray =
      cuda_make_ray(cuda_vec3_subtract(ray.origin, translate.offset),
                    ray.direction, ray.time);

  // Dispatch the hit function of the wrapped object.
  if (!cuda_hittable_hit(*translate.object, moved_ray, t_range, rec,
                         rand_state))
    return false;

  // Offset the hit point back to world space.
  rec.point = cuda_vec3_add(rec.point, translate.offset);

  return true;
}

__device__ double cuda_translate_pdf_value(const CudaTranslate &translate,
                                           const CudaPoint3 &origin,
                                           const CudaVec3 &direction) {
  // Move the origin backwards by the translation offset.
  return cuda_hittable_pdf_value(*translate.object,
                                 cuda_vec3_subtract(origin, translate.offset),
                                 direction);
}

__device__ CudaVec3 cuda_translate_random(const CudaTranslate &translate,
                                          const CudaPoint3 &origin,
                                          curandState *state) {
  // Move the origin backwards by the translation offset.
  return cuda_hittable_random(
      *translate.object, cuda_vec3_subtract(origin, translate.offset), state);
}

// JSON serialization function for CudaTranslate.
std::string cuda_json_translate(const CudaTranslate &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaTranslate\",";
  oss << "\"address\":\"" << &obj << "\",";
  if (obj.object) {
    CudaHittable host_object;
    cudaMemcpyDeviceToHostSafe(&host_object, obj.object, 1);
    oss << "\"object\":" << cuda_json_hittable(host_object) << ",";
  } else {
    oss << "\"object\":null,";
  }
  oss << "\"offset\":" << cuda_json_vec3(obj.offset) << ",";
  oss << "\"bbox\":" << cuda_json_aabb(obj.bbox);
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA