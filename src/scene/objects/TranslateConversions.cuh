#pragma once

#ifdef USE_CUDA

#include "../../optimization/AABBConversions.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "Translate.cuh"

// Forward declarations.
class Translate;
struct CudaHittable;
class Hittable;

// Forward declaration of conversion functions.
inline CudaHittable cpu_to_cuda_hittable(const HittablePtr &cpu_hittable);
inline HittablePtr cuda_to_cpu_hittable(const CudaHittable &cuda_hittable);

// Convert CPU Translate to CUDA Translate.
inline CudaTranslate cpu_to_cuda_translate(const Translate &cpu_translate) {
  // Extract properties from CPU Translate using getters.
  auto cpu_object = cpu_translate.get_object();
  Vec3 cpu_offset = cpu_translate.get_offset();

  // Convert object to CUDA format.
  CudaHittable *cuda_object = new CudaHittable();
  *cuda_object = cpu_to_cuda_hittable(cpu_object);

  // Convert offset.
  CudaVec3 cuda_offset = cpu_to_cuda_vec3(cpu_offset);

  // Create CUDA translate.
  return CudaTranslate(cuda_object, cuda_offset);
}

// Convert CUDA Translate to CPU Translate.
inline Translate cuda_to_cpu_translate(const CudaTranslate &cuda_translate) {
  // Convert object back to CPU format.
  auto cpu_object = cuda_to_cpu_hittable(*cuda_translate.object);

  // Convert offset back to CPU format.
  Vec3 cpu_offset = cuda_to_cpu_vec3(cuda_translate.offset);

  // Create CPU translate.
  return Translate(cpu_object, cpu_offset);
}

// Create CUDA translate from hittable and offset.
__host__ __device__ inline CudaTranslate
create_cuda_translate(const CudaHittable *object, const CudaVec3 &offset) {
  return CudaTranslate(object, offset);
}

// Helper function to create translated sphere.
__host__ __device__ inline CudaTranslate
create_cuda_translated_sphere(const CudaPoint3 &center, double radius,
                              const CudaMaterial &material,
                              const CudaVec3 &offset) {
  CudaHittable *sphere_obj = new CudaHittable();
  sphere_obj->type = HITTABLE_SPHERE;
  sphere_obj->sphere = create_cuda_sphere_static(center, radius, material);

  return CudaTranslate(sphere_obj, offset);
}

// Helper function to create translated plane.
__host__ __device__ inline CudaTranslate
create_cuda_translated_plane(const CudaPoint3 &corner, const CudaVec3 &u,
                             const CudaVec3 &v, const CudaMaterial &material,
                             const CudaVec3 &offset) {
  CudaHittable *plane_obj = new CudaHittable();
  plane_obj->type = HITTABLE_PLANE;
  plane_obj->plane = create_cuda_plane(corner, u, v, material);

  return CudaTranslate(plane_obj, offset);
}

// Memory management for translate objects
inline void cleanup_cuda_translate(CudaTranslate &cuda_translate) {
  if (cuda_translate.object) {
    delete cuda_translate.object;
    cuda_translate.object = nullptr;
  }
}

#endif // USE_CUDA