#pragma once

#ifdef USE_CUDA

#include "Utility.cuh"
#include <curand_kernel.h>

// POD vector type optimized for CUDA.
struct CudaVec3 {
  double x, y, z;
};

// Vec3 initialization functions.
__host__ __device__ inline CudaVec3 cuda_make_vec3(double x, double y,
                                                   double z) {
  CudaVec3 v;
  v.x = x;
  v.y = y;
  v.z = z;
  return v;
}

// Vec3 utility functions.
__host__ __device__ inline CudaVec3 cuda_vec3_negate(const CudaVec3 &v) {
  return cuda_make_vec3(-v.x, -v.y, -v.z);
}

__host__ __device__ inline double cuda_vec3_get(const CudaVec3 &v, int i) {
  return i == 0 ? v.x : (i == 1 ? v.y : v.z);
}

__host__ __device__ inline void cuda_vec3_set(CudaVec3 &v, int i, double val) {
  if (i == 0)
    v.x = val;
  else if (i == 1)
    v.y = val;
  else
    v.z = val;
}

__host__ __device__ inline CudaVec3 cuda_vec3_add(const CudaVec3 &a,
                                                  const CudaVec3 &b) {
  return cuda_make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline CudaVec3 cuda_vec3_subtract(const CudaVec3 &a,
                                                       const CudaVec3 &b) {
  return cuda_make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline CudaVec3 cuda_vec3_multiply_scalar(const CudaVec3 &v,
                                                              double t) {
  return cuda_make_vec3(v.x * t, v.y * t, v.z * t);
}

__host__ __device__ inline CudaVec3 cuda_vec3_divide_scalar(const CudaVec3 &v,
                                                            double t) {
  return cuda_vec3_multiply_scalar(v, 1.0 / t);
}

__host__ __device__ inline double cuda_vec3_length(const CudaVec3 &v) {
  return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ inline double cuda_vec3_length_squared(const CudaVec3 &v) {
  return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ inline bool cuda_vec3_near_zero(const CudaVec3 &v) {
  const double s = 1e-8;
  return fabs(v.x) < s && fabs(v.y) < s && fabs(v.z) < s;
}

__device__ inline CudaVec3 cuda_vec3_random(curandState *state) {
  return cuda_make_vec3(cuda_random_double(state), cuda_random_double(state),
                        cuda_random_double(state));
}

__device__ inline CudaVec3 cuda_vec3_random(double min, double max,
                                            curandState *state) {
  return cuda_make_vec3(cuda_random_double(state, min, max),
                        cuda_random_double(state, min, max),
                        cuda_random_double(state, min, max));
}

#endif // USE_CUDA
