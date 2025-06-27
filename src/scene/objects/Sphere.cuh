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
#include <iomanip>
#include <sstream>

// POD struct representing a sphere hittable object.
struct CudaSphere {
  CudaRay center;
  double radius;
  size_t material_index;
  CudaAABB bbox;
};

// Sphere initialization functions.

__host__ __device__ inline CudaSphere cuda_make_sphere(const CudaRay &center,
                                                       double radius,
                                                       size_t material_index,
                                                       const CudaAABB &bbox) {
  CudaSphere sphere;
  sphere.center = center;
  sphere.radius = fmax(0.0, radius);
  sphere.material_index = material_index;
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

__host__ __device__ inline CudaAABB
cuda_sphere_get_bounding_box(const CudaSphere &sphere) {
  return sphere.bbox;
}

// JSON serialization function for CudaSphere.
inline std::string cuda_json_sphere(const CudaSphere &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaSphere\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"center\":" << cuda_json_ray(obj.center) << ",";
  oss << "\"radius\":" << obj.radius << ",";
  oss << "\"material_index\":" << obj.material_index << ",";
  oss << "\"bbox\":" << cuda_json_aabb(obj.bbox);
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA
