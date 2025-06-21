#pragma once

#ifdef USE_CUDA

#include "../../core/RayConversions.cuh"
#include "../../optimization/AABBConversions.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "../materials/MaterialConversions.cuh"
#include "Sphere.cuh"
#include "Sphere.hpp"

// Convert CPU Sphere to CUDA Sphere.
inline CudaSphere cpu_to_cuda_sphere(const Sphere &cpu_sphere) {
  // Extract sphere properties using public getter methods.
  Ray cpu_center = cpu_sphere.get_center();
  double radius = cpu_sphere.get_radius();
  MaterialPtr cpu_material = cpu_sphere.get_material();

  // Convert center and material to CUDA format.
  CudaRay cuda_center = cpu_to_cuda_ray(cpu_center);
  CudaMaterial cuda_material = cpu_to_cuda_material(*cpu_material);

  CudaAABB cuda_bbox = cpu_to_cuda_aabb(cpu_sphere.get_bounding_box());

  return CudaSphere(cuda_center, radius, &cuda_material, cuda_bbox);
}

// Convert CUDA Sphere to CPU Sphere.
inline std::shared_ptr<Sphere>
cuda_to_cpu_sphere(const CudaSphere &cuda_sphere) {
  // Extract center at time 0.
  CudaPoint3 cuda_center = cuda_sphere.center.at(0.0);
  Point3 cpu_center = cuda_to_cpu_vec3(cuda_center);

  // Convert material.
  MaterialPtr cpu_material = cuda_to_cpu_material(*cuda_sphere.material);

  AABB cpu_bbox = cuda_to_cpu_aabb(cuda_sphere.bbox);

  // Create CPU sphere.
  return std::make_shared<Sphere>(cpu_center, cuda_sphere.radius, cpu_material);
}

#endif // USE_CUDA