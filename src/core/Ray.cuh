#pragma once

#ifdef USE_CUDA

#include "../utils/math/Vec3.cuh"
#include "Vec3Types.cuh"

// Represents a light ray in CUDA, using the parametric equation:
// point = origin + t * direction. Time parameter `t` tracks distance
// along the ray, useful for motion blur or time-varying scenes.
struct CudaRay {
  CudaPoint3 origin;
  CudaVec3 direction;
  double time;
};

// Initializes a ray with default zero origin, direction, and time.
__device__ __forceinline__ CudaRay cuda_make_ray() {
  CudaRay ray;
  ray.origin = CudaPoint3(0.0, 0.0, 0.0);
  ray.direction = CudaVec3(0.0, 0.0, 0.0);
  ray.time = 0.0;
  return ray;
}

// Initializes a ray with specified origin and direction, default time = 0.
__device__ __forceinline__ CudaRay cuda_make_ray(const CudaPoint3 &origin,
                                                 const CudaVec3 &direction) {
  CudaRay ray;
  ray.origin = origin;
  ray.direction = direction;
  ray.time = 0.0;
  return ray;
}

// Initializes a ray with specified origin, direction, and time.
__device__ __forceinline__ CudaRay cuda_make_ray(const CudaPoint3 &origin,
                                                 const CudaVec3 &direction,
                                                 double time) {
  CudaRay ray;
  ray.origin = origin;
  ray.direction = direction;
  ray.time = time;
  return ray;
}

// Returns the point at parameter t: origin + t * direction.
__device__ __forceinline__ CudaPoint3 cuda_ray_at(const CudaRay &ray,
                                                  double t) {
  return ray.origin + t * ray.direction;
}

// Accessors
__device__ __forceinline__ const CudaPoint3 &
cuda_ray_origin(const CudaRay &ray) {
  return ray.origin;
}

__device__ __forceinline__ const CudaVec3 &
cuda_ray_direction(const CudaRay &ray) {
  return ray.direction;
}

__device__ __forceinline__ double cuda_ray_time(const CudaRay &ray) {
  return ray.time;
}

#endif // USE_CUDA
