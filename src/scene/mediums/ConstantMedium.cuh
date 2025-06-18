#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Interval.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.hpp"
#include "../hittables/CudaHittable.cuh"
#include "CudaMaterial.cuh"
#include "CudaTexture.cuh"
#include <curand_kernel.h>

// Represents a volume medium like fog/smoke using exponential attenuation.
struct CudaConstantMedium {
  const CudaHittable *boundary; // Underlying geometry
  double neg_inv_density;       // -1 / density (precomputed)
  CudaMaterial phase_function;  // Isotropic scattering material
};

// Handles ray-medium interaction: scattering inside a constant density medium.
__device__ bool cuda_constant_medium_hit(const CudaConstantMedium &medium,
                                         const CudaRay &ray,
                                         CudaInterval t_values,
                                         CudaHitRecord &record,
                                         curandState *rand_state) {

  CudaHitRecord rec1, rec2;

  if (!cuda_hittable_hit(*medium.boundary, ray, CUDA_UNIVERSE_INTERVAL, rec1,
                         rand_state))
    return false;

  if (!cuda_hittable_hit(*medium.boundary, ray,
                         CudaInterval(rec1.t + 0.0001, CUDA_INF), rec2,
                         rand_state))
    return false;

  if (rec1.t < t_values.min())
    rec1.t = t_values.min();
  if (rec2.t > t_values.max())
    rec2.t = t_values.max();
  if (rec1.t >= rec2.t)
    return false;
  if (rec1.t < 0)
    rec1.t = 0;

  double ray_length = cuda_length(ray.direction);
  double distance_inside = (rec2.t - rec1.t) * ray_length;

  double hit_distance =
      medium.neg_inv_density * log(curand_uniform_double(rand_state));

  if (hit_distance > distance_inside)
    return false;

  record.t = rec1.t + hit_distance / ray_length;
  record.point = cuda_ray_at(ray, record.t);
  record.normal = CudaVec3(1, 0, 0); // arbitrary
  record.front_face = true;
  record.material = medium.phase_function;

  return true;
}

#endif // USE_CUDA
