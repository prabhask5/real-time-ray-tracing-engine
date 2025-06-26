#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../utils/math/Interval.cuh"
#include "../../utils/math/Utility.cuh"
#include "../../utils/math/Vec3Utility.cuh"
#include "../materials/Material.cuh"
#include <curand_kernel.h>

// POD struct for participating medium (smoke, fog, gas).
struct CudaConstantMedium {
  const CudaHittable *boundary;
  double neg_inv_density;
  const CudaMaterial *phase_function;
};

// Constant medium initialization functions.
__device__ inline CudaConstantMedium
cuda_make_constant_medium(const CudaHittable *boundary, double density,
                          const CudaMaterial *phase_function) {
  CudaConstantMedium medium;
  medium.boundary = boundary;
  medium.neg_inv_density = -1.0 / density;
  medium.phase_function = phase_function;
  return medium;
}

__device__ CudaConstantMedium cuda_make_constant_medium(
    const CudaHittable *boundary, double density, CudaTexture *texture);

// Constant medium utility functions.
__device__ bool cuda_constant_medium_hit(const CudaConstantMedium &medium,
                                         const CudaRay &ray,
                                         CudaInterval t_range,
                                         CudaHitRecord &rec,
                                         curandState *rand_state);

__device__ CudaAABB
cuda_constant_medium_get_bounding_box(const CudaConstantMedium &medium);

#endif // USE_CUDA
