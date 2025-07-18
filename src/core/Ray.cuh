#pragma once

#ifdef USE_CUDA

#include "../utils/math/Vec3.cuh"
#include "../utils/math/Vec3Utility.cuh"
#include "Vec3Types.cuh"
#include <iomanip>
#include <sstream>

// POD struct representing a light ray.
struct CudaRay {
  CudaPoint3 origin;
  CudaVec3 direction;
  double time;
};

// Ray initialization functions.
__host__ __device__ inline CudaRay cuda_make_ray(const CudaPoint3 &origin,
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

// JSON serialization function for CudaRay.
inline std::string cuda_json_ray(const CudaRay &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaRay\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"origin\":" << cuda_json_vec3(obj.origin) << ",";
  oss << "\"direction\":" << cuda_json_vec3(obj.direction) << ",";
  oss << "\"time\":" << obj.time;
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA
