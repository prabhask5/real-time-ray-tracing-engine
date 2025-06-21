#pragma once

#ifdef USE_CUDA

#include "Utility.cuh"
#include <curand_kernel.h>

// POD vector type optimized for CUDA.
struct CudaVec3 {
  double x, y, z;

  __device__ CudaVec3() : x(0), y(0), z(0) {}

  __host__ __device__ CudaVec3(double x_, double y_, double z_)
      : x(x_), y(y_), z(z_) {}

  // Operator overloads for 3D vector.

  __device__ CudaVec3 operator-() const { return CudaVec3(-x, -y, -z); }

  __device__ double operator[](int i) const {
    return i == 0 ? x : (i == 1 ? y : z);
  }

  __device__ double &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }

  __device__ CudaVec3 &operator+=(const CudaVec3 &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }

  __device__ CudaVec3 &operator-=(const CudaVec3 &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }

  __device__ CudaVec3 &operator*=(double t) {
    x *= t;
    y *= t;
    z *= t;
    return *this;
  }

  __device__ CudaVec3 &operator/=(double t) { return *this *= 1.0 / t; }

  // Complex getter methods.

  __device__ double length() const { return sqrt(x * x + y * y + z * z); }

  __device__ double length_squared() const { return x * x + y * y + z * z; }

  // Return true if the vector is close to zero in all dimensions.
  __device__ bool near_zero() const {
    const double s = 1e-8;
    return fabs(x) < s && fabs(y) < s && fabs(z) < s;
  }
};

__device__ inline CudaVec3 cuda_vec_random(curandState *state) {
  return CudaVec3(cuda_random_double(state), cuda_random_double(state),
                  cuda_random_double(state));
}

__device__ inline CudaVec3 cuda_vec_random(double min, double max,
                                           curandState *state) {
  return CudaVec3(cuda_random_double(state, min, max),
                  cuda_random_double(state, min, max),
                  cuda_random_double(state, min, max));
}

#endif // USE_CUDA
