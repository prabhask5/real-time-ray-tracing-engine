#pragma once

#ifdef USE_CUDA

#include "../utils/math/Vec3Conversions.cuh"
#include "Ray.cuh"
#include "Ray.hpp"

// Convert CPU Ray to CUDA Ray
inline CudaRay cpu_to_cuda_ray(const Ray &cpu_ray) {
  return CudaRay(cpu_to_cuda_vec3(cpu_ray.origin()),
                 cpu_to_cuda_vec3(cpu_ray.direction()), cpu_ray.time());
}

// Convert CUDA Ray to CPU Ray
inline Ray cuda_to_cpu_ray(const CudaRay &cuda_ray) {
  return Ray(cuda_to_cpu_vec3(cuda_ray.origin),
             cuda_to_cpu_vec3(cuda_ray.direction), cuda_ray.time);
}

// Batch conversion functions for performance
void batch_cpu_to_cuda_ray(const Ray *cpu_rays, CudaRay *cuda_rays, int count);
void batch_cuda_to_cpu_ray(const CudaRay *cuda_rays, Ray *cpu_rays, int count);

#endif // USE_CUDA