#pragma once

#ifdef USE_CUDA

#include "../utils/math/Vec3.cuh"
#include "Vec3Types.cuh"

// Represents a light ray in CUDA, using the parametric equation:
// point = origin + t * direction. Time parameter `t` tracks distance
// along the ray, useful for motion blur or time-varying scenes.
struct CudaRay {
  CudaPoint3 origin;
  CudaVec3 direction;
  double time;

  // Initializes a ray with default zero origin, direction, and time.
  __device__ CudaRay()
      : origin(CudaPoint3(0.0, 0.0, 0.0)), direction(CudaVec3(0.0, 0.0, 0.0)),
        time(0.0) {}

  // Initializes a ray with specified origin and direction, default time = 0.
  __device__ CudaRay(const CudaPoint3 &_origin, const CudaVec3 &_direction)
      : origin(_origin), direction(_direction), time(0.0) {}

  // Initializes a ray with specified origin, direction, and time.
  __device__ CudaRay(const CudaPoint3 &_origin, const CudaVec3 &_direction,
                     double _time)
      : origin(_origin), direction(_direction), time(_time) {}

  // Returns the point at parameter t: origin + t * direction.
  __device__ __forceinline__ CudaPoint3 at(double t) const {
    return origin + t * direction;
  }
};

#endif // USE_CUDA
