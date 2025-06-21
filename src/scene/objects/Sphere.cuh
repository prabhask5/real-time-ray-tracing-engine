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

// CUDA-compatible sphere with motion blur support.
// Represents a sphere hittable object.
struct CudaSphere {
  CudaRay center;
  double radius;
  const CudaMaterial *material;
  CudaAABB bbox;

  __device__ CudaSphere() {} // Default constructor.

  // Initialize sphere with static center.
  __device__ CudaSphere(const CudaPoint3 &_center, double _radius,
                        const CudaMaterial *_material);

  // Initialize sphere with moving center (motion blur).
  __device__ CudaSphere(const CudaPoint3 &before_center,
                        const CudaPoint3 &after_center, double _radius,
                        const CudaMaterial *_material);

  // Initialize sphere from direct members.
  __host__ __device__ CudaSphere(const CudaRay &_center, double _radius,
                                 const CudaMaterial *_material,
                                 const CudaAABB _bbox)
      : center(_center), radius(fmax(0.0, _radius)), material(_material),
        bbox(_bbox) {}

  // Hit test for sphere.
  __device__ bool hit(const CudaRay &ray, CudaInterval t_range,
                      CudaHitRecord &rec, curandState *rand_state) const;

  // PDF for sampling the sphere (solid angle).
  __device__ double pdf_value(const CudaPoint3 &origin,
                              const CudaVec3 &direction) const;

  // Importance sample a direction toward the sphere.
  __device__ CudaVec3 random(const CudaPoint3 &origin,
                             curandState *rand_state) const;

  // Get bounding box for the sphere.
  __device__ inline CudaAABB get_bounding_box() const { return bbox; }
};

#endif // USE_CUDA
