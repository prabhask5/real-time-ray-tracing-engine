#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../optimization/AABB.cuh"
#include "../../utils/math/Interval.cuh"
#include "../../utils/math/ONB.cuh"
#include "../../utils/math/Vec3Utility.cuh"
#include "../materials/Material.cuh"
#include <curand_kernel.h>

// POD struct representing a sphere hittable object.
struct CudaSphere {
  CudaRay center;
  double radius;
  const CudaMaterial *material;
  CudaAABB bbox;
};

// Sphere initialization functions.

__device__ inline CudaSphere cuda_make_sphere(const CudaPoint3 &center,
                                              double radius,
                                              const CudaMaterial *material);

__device__ inline CudaSphere cuda_make_sphere(const CudaPoint3 &before_center,
                                              const CudaPoint3 &after_center,
                                              double radius,
                                              const CudaMaterial *material);

__host__ __device__ inline CudaSphere
cuda_make_sphere(const CudaRay &center, double radius,
                 const CudaMaterial *material, const CudaAABB &bbox) {
  CudaSphere sphere;
  sphere.center = center;
  sphere.radius = fmax(0.0, radius);
  sphere.material = material;
  sphere.bbox = bbox;
  return sphere;
}

// Sphere utility functions.
__device__ bool cuda_sphere_hit(const CudaSphere &sphere, const CudaRay &ray,
                                CudaInterval t_range, CudaHitRecord &rec,
                                curandState *rand_state);

__device__ double cuda_sphere_pdf_value(const CudaSphere &sphere,
                                        const CudaPoint3 &origin,
                                        const CudaVec3 &direction);

__device__ CudaVec3 cuda_sphere_random(const CudaSphere &sphere,
                                       const CudaPoint3 &origin,
                                       curandState *rand_state);

__device__ inline CudaAABB
cuda_sphere_get_bounding_box(const CudaSphere &sphere) {
  return sphere.bbox;
}

#endif // USE_CUDA
