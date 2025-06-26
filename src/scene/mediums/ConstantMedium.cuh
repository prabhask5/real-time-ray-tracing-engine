#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../utils/math/Interval.cuh"
#include "../../utils/math/Utility.cuh"
#include "../../utils/math/Vec3Utility.cuh"
#include "../../utils/memory/CudaSceneContext.cuh"
#include "../materials/Material.cuh"
#include <curand_kernel.h>

// POD struct for participating medium (smoke, fog, gas).
struct CudaConstantMedium {
  const CudaHittable *boundary;
  double neg_inv_density;
  size_t phase_function_index;
};

// Constant medium initialization functions.
inline CudaConstantMedium
cuda_make_constant_medium(const CudaHittable *boundary, double density,
                          size_t phase_function_index) {
  CudaConstantMedium medium;
  medium.boundary = boundary;
  medium.neg_inv_density = -1.0 / density;
  medium.phase_function_index = phase_function_index;
  return medium;
}

// Constant medium utility functions.
__device__ bool cuda_constant_medium_hit(const CudaConstantMedium &medium,
                                         const CudaRay &ray,
                                         CudaInterval t_range,
                                         CudaHitRecord &rec,
                                         curandState *rand_state);

__host__ __device__ CudaAABB
cuda_constant_medium_get_bounding_box(const CudaConstantMedium &medium);

#endif // USE_CUDA
