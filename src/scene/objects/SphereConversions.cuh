#pragma once

#ifdef USE_CUDA

#include "../../core/RayConversions.cuh"
#include "../../optimization/AABBConversions.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "../materials/MaterialConversions.cuh"
#include "Sphere.cuh"
#include "Sphere.hpp"

// Convert CPU Sphere to CUDA Sphere POD struct.
inline CudaSphere cpu_to_cuda_sphere(const Sphere &cpu_sphere) {
  Ray cpu_center = cpu_sphere.get_center();
  double radius = cpu_sphere.get_radius();

  CudaRay cuda_center = cpu_to_cuda_ray(cpu_center);
  CudaAABB cuda_bbox = cpu_to_cuda_aabb(cpu_sphere.get_bounding_box());

  // Leave material blank for now.
  return cuda_make_sphere(cuda_center, radius, nullptr, cuda_bbox);
}

#endif // USE_CUDA