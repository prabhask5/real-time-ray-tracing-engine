#pragma once

#ifdef USE_CUDA

#include "../../optimization/AABBConversions.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "Translate.cuh"
#include "Translate.hpp"

// Forward declaration for hittable conversion.
struct CudaHittable;
CudaHittable cpu_to_cuda_hittable(const Hittable &cpu_hittable);
HittablePtr cuda_to_cpu_hittable(const CudaHittable &cuda_hittable);

// Convert CPU Translate to CUDA Translate.
inline CudaTranslate cpu_to_cuda_translate(const Translate &cpu_translate) {
  // Extract properties from CPU Translate using getters.
  HittablePtr cpu_object = cpu_translate.get_object();
  Vec3 cpu_offset = cpu_translate.get_offset();

  // Convert object to CUDA format.
  CudaHittable *cuda_object = new CudaHittable();
  *cuda_object = cpu_to_cuda_hittable(*cpu_object);

  // Convert offset.
  CudaVec3 cuda_offset = cpu_to_cuda_vec3(cpu_offset);

  CudaAABB cuda_bbox = cpu_to_cuda_aabb(cpu_translate.get_bounding_box());

  // Create CUDA translate.
  return CudaTranslate(cuda_object, cuda_offset, cuda_bbox);
}

// Convert CUDA Translate to CPU Translate.
inline Translate cuda_to_cpu_translate(const CudaTranslate &cuda_translate) {
  // Convert object back to CPU format.
  HittablePtr cpu_object = cuda_to_cpu_hittable(*cuda_translate.object);

  // Convert offset back to CPU format.
  Vec3 cpu_offset = cuda_to_cpu_vec3(cuda_translate.offset);

  AABB cpu_bbox = cuda_to_cpu_aabb(cuda_translate.bbox);

  // Create CPU translate.
  return Translate(cpu_object, cpu_offset, cpu_bbox);
}

// Memory management for translate objects.
inline void cleanup_cuda_translate(CudaTranslate &cuda_translate) {
  if (cuda_translate.object) {
    delete cuda_translate.object;
    cuda_translate.object = nullptr;
  }
}

#endif // USE_CUDA