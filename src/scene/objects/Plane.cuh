#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../optimization/AABB.cuh"
#include "../../utils/math/Interval.cuh"
#include "../../utils/math/Vec3Utility.cuh"
#include "../materials/Material.cuh"
#include <curand_kernel.h>

// A planar parallelogram with two edge vectors and a corner point.
struct CudaPlane {
  CudaPoint3 corner;
  CudaVec3 u_side;
  CudaVec3 v_side;
  CudaVec3 w; // Used for computing UV coordinates
  CudaVec3 normal;
  double D;
  double surface_area;
  CudaAABB bbox;
  const CudaMaterial *material;

  __device__ CudaPlane() {} // Default constructor.

  // Initialize CUDA plane.
  __device__ CudaPlane(const CudaPoint3 &_corner, const CudaVec3 &_u_side,
                       const CudaVec3 &_v_side, const CudaMaterial *_material);

  // Computes if a ray hits a CUDA plane, and writes to record if so.
  __device__ bool hit(const CudaRay &ray, CudaInterval t_values,
                      CudaHitRecord &record, curandState *rand_state) const;

  // Computes the PDF value of sampling the plane from a point along a
  // direction.
  __device__ double pdf_value(const CudaPoint3 &origin,
                              const CudaVec3 &direction) const;

  // Samples a random direction toward a point on the plane.
  __device__ CudaVec3 random(const CudaPoint3 &origin,
                             curandState *rand_state) const;

  // Get bounding box for the plane.
  __device__ inline CudaAABB get_bounding_box() const { return bbox; }
};

#endif // USE_CUDA
