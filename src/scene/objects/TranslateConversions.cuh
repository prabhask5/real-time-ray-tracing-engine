#pragma once

#ifdef USE_CUDA

#include "../../optimization/AABBConversions.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "Translate.cuh"
#include "Translate.hpp"

// Forward declaration for hittable conversion.
struct CudaHittable;
CudaHittable cpu_to_cuda_hittable(const Hittable &cpu_hittable);

// Convert CPU Translate to CUDA Translate POD struct.
inline CudaTranslate cpu_to_cuda_translate(const Translate &cpu_translate) {
  HittablePtr cpu_object = cpu_translate.get_object();
  Vec3 cpu_offset = cpu_translate.get_offset();

  CudaHittable *cuda_object = new CudaHittable();
  *cuda_object = cpu_to_cuda_hittable(*cpu_object);

  CudaVec3 cuda_offset = cpu_to_cuda_vec3(cpu_offset);
  CudaAABB cuda_bbox = cpu_to_cuda_aabb(cpu_translate.get_bounding_box());

  return cuda_make_translate(cuda_object, cuda_offset, cuda_bbox);
}

#endif // USE_CUDA