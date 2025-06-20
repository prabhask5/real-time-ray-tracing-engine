#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Hittable.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../utils/math/Interval.cuh"
#include "../../utils/math/Utility.cuh"
#include "../../utils/math/Vec3Utility.cuh"
#include "../materials/Material.cuh"
#include <curand_kernel.h>

// Represents a volume medium like fog/smoke using exponential attenuation.
struct CudaConstantMedium {
  const CudaHittable *boundary; // Underlying geometry
  double neg_inv_density;       // -1 / density (precomputed)
  CudaMaterial phase_function;  // Isotropic scattering material

  __device__ CudaConstantMedium(const CudaHittable *_boundary, double density,
                                const CudaMaterial &_phase_function)
      : boundary(_boundary), neg_inv_density(-1.0 / density),
        phase_function(_phase_function) {}

  // Handles ray-medium interaction: scattering inside a constant density
  // medium.
  __device__ bool hit(const CudaRay &ray, CudaInterval t_values,
                      CudaHitRecord &record, curandState *rand_state) {

    CudaHitRecord rec1, rec2;

    if (!boundary->hit(ray, CudaInterval(-CUDA_INF, CUDA_INF), rec1,
                       rand_state))
      return false;

    if (!boundary->hit(ray, CudaInterval(rec1.t + 0.0001, CUDA_INF), rec2,
                       rand_state))
      return false;

    if (rec1.t < t_values.min)
      rec1.t = t_values.min;
    if (rec2.t > t_values.max)
      rec2.t = t_values.max;
    if (rec1.t >= rec2.t)
      return false;
    if (rec1.t < 0)
      rec1.t = 0;

    double ray_length = ray.direction.length();
    double distance_inside = (rec2.t - rec1.t) * ray_length;

    double hit_distance = neg_inv_density * log(cuda_random_double(rand_state));

    if (hit_distance > distance_inside)
      return false;

    record.t = rec1.t + hit_distance / ray_length;
    record.point = ray.at(record.t);
    record.normal = CudaVec3(1, 0, 0); // arbitrary
    record.front_face = true;
    record.material_type = phase_function.type;
    record.material_data = (void *)&phase_function;

    return true;
  }

  // Get bounding box for constant medium (uses boundary)
  __device__ inline CudaAABB get_bounding_box() {
    return boundary->get_bounding_box();
  }
};

#endif // USE_CUDA
