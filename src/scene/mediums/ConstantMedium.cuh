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

// Represents a participating medium—such as smoke, fog, or gas—that can scatter
// rays randomly inside a volume instead of just reflecting off surfaces. It
// wraps another hittable object (usually a box or sphere) and simulates light
// scattering inside it.
struct CudaConstantMedium {
  const CudaHittable *boundary;       // Underlying geometry
  double neg_inv_density;             // -1 / density (precomputed)
  const CudaMaterial *phase_function; // Isotropic scattering material

  __device__ CudaConstantMedium() {} // Default constructor.

  __device__ CudaConstantMedium(const CudaHittable *_boundary, double density,
                                CudaTexture *texture);

  __host__ __device__ CudaConstantMedium(const CudaHittable *_boundary,
                                         double density,
                                         const CudaMaterial *_phase_function)
      : boundary(_boundary), neg_inv_density(-1.0 / density),
        phase_function(_phase_function) {}

  // Handles ray interaction with volumetric media—like fog, smoke, or clouds—by
  // determining whether a ray randomly scatters inside the medium instead of
  // just passing through it. It's used in ray tracing to simulate participating
  // media.
  __device__ bool hit(const CudaRay &ray, CudaInterval t_values,
                      CudaHitRecord &record, curandState *rand_state) const;

  // Get bounding box for constant medium (uses boundary).
  __device__ CudaAABB get_bounding_box() const;
};

#endif // USE_CUDA
