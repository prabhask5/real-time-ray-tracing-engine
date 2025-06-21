#ifdef USE_CUDA

#include "../../core/Hittable.cuh"
#include "Translate.cuh"

__device__ CudaTranslate::CudaTranslate(const CudaHittable *_object,
                                        const CudaVec3 &_offset)
    : object(_object), offset(_offset) {
  bbox = object->get_bounding_box() + offset;
}

__device__ bool CudaTranslate::hit(const CudaRay &ray, CudaInterval t_range,
                                   CudaHitRecord &rec,
                                   curandState *rand_state) const {

  // Move the ray backwards by the translation offset.
  CudaRay moved_ray = CudaRay(ray.origin - offset, ray.direction, ray.time);

  // Dispatch the hit function of the wrapped object.
  if (!object->hit(moved_ray, t_range, rec, rand_state))
    return false;

  // Offset the hit point back to world space.
  rec.point += offset;

  return true;
}

__device__ double CudaTranslate::pdf_value(const CudaPoint3 &origin,
                                           const CudaVec3 &direction) const {
  // Move the origin backwards by the translation offset.
  return object->pdf_value(origin - offset, direction);
}

__device__ CudaVec3 CudaTranslate::random(const CudaPoint3 &origin,
                                          curandState *state) const {
  // Move the origin backwards by the translation offset.
  return object->random(origin - offset, state);
}

#endif // USE_CUDA