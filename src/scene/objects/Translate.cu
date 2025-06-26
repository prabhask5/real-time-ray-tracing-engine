#ifdef USE_CUDA

#include "../../core/Hittable.cuh"
#include "Translate.cuh"

CudaTranslate cuda_make_translate(const CudaHittable *object,
                                  const CudaVec3 &offset) {
  CudaTranslate translate;
  translate.object = object;
  translate.offset = offset;
  CudaAABB object_bbox = cuda_hittable_get_bounding_box(*object);
  translate.bbox =
      cuda_make_aabb(object_bbox.x + offset.x, object_bbox.y + offset.y,
                     object_bbox.z + offset.z);
  return translate;
}

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

#endif // USE_CUDA