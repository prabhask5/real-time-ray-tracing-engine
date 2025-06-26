#pragma once

#ifdef USE_CUDA

#include "../../core/Vec3Types.cuh"
#include "Vec3Utility.cuh"

// POD struct representing an orthonormal basis.
struct CudaONB {
  CudaVec3 axis[3];
};

// ONB initialization functions.

// Constructs an ONB from a normal vector.
__device__ inline CudaONB cuda_make_onb(const CudaVec3 &n) {
  CudaONB onb;
  onb.axis[2] = cuda_vec3_unit_vector(n);
  CudaVec3 a = (fabs(onb.axis[2].x) > 0.9) ? cuda_make_vec3(0, 1, 0)
                                           : cuda_make_vec3(1, 0, 0);
  onb.axis[1] = cuda_vec3_unit_vector(cuda_vec3_cross_product(onb.axis[2], a));
  onb.axis[0] = cuda_vec3_cross_product(onb.axis[2], onb.axis[1]);
  return onb;
}

// ONB utility functions.
__device__ inline const CudaVec3 &cuda_onb_u(const CudaONB &onb) {
  return onb.axis[0];
}
__device__ inline const CudaVec3 &cuda_onb_v(const CudaONB &onb) {
  return onb.axis[1];
}
__device__ inline const CudaVec3 &cuda_onb_w(const CudaONB &onb) {
  return onb.axis[2];
}

// Transforms a vector from local space to world space using this ONB.
__device__ inline CudaVec3 cuda_onb_transform(const CudaONB &onb,
                                              const CudaVec3 &v) {
  return cuda_vec3_add(
      cuda_vec3_add(
          cuda_vec3_multiply_scalar(onb.axis[0], cuda_vec3_get(v, 0)),
          cuda_vec3_multiply_scalar(onb.axis[1], cuda_vec3_get(v, 1))),
      cuda_vec3_multiply_scalar(onb.axis[2], cuda_vec3_get(v, 2)));
}

#endif // USE_CUDA