#pragma once

#ifdef USE_CUDA

#include "../utils/math/Vec3Conversions.cuh"
#include "Ray.cuh"
#include "Ray.hpp"

// Convert CPU Ray to CUDA Ray POD struct.
inline CudaRay cpu_to_cuda_ray(const Ray &cpu_ray) {
  return cuda_make_ray(cpu_to_cuda_vec3(cpu_ray.origin()),
                       cpu_to_cuda_vec3(cpu_ray.direction()), cpu_ray.time());
}

#endif // USE_CUDA