#pragma once

#ifdef USE_CUDA

#include "../../core/Vec3Types.cuh"
#include "Vec3Utility.cuh"

// This CudaONB (orthonormal basis) struct constructs and manages a local
// coordinate system — often used in ray tracing to convert vectors between
// local and world space, particularly for sampling directions relative to a
// surface normal.
struct CudaONB {
  CudaVec3 axis[3];

  // Default constructor.
  __device__ CudaONB() {}

  // Constructs an ONB from a normal vector.
  // We use n as the z-axis, and compute cross-products to find the x and y-axes
  // (they need to be perpendicular).
  __device__ CudaONB(const CudaVec3 &n) {
    axis[2] = cuda_unit_vector(n);
    CudaVec3 a =
        (fabs(axis[2].x) > 0.9) ? CudaVec3(0, 1, 0) : CudaVec3(1, 0, 0);
    axis[1] = cuda_unit_vector(cuda_cross_product(axis[2], a));
    axis[0] = cuda_cross_product(axis[2], axis[1]);
  }

  // Getter const methods.

  __device__ const CudaVec3 &u() const { return axis[0]; }
  __device__ const CudaVec3 &v() const { return axis[1]; }
  __device__ const CudaVec3 &w() const { return axis[2]; }

  // Transforms a vector from local space to world space using this ONB.
  __device__ CudaVec3 transform(const CudaVec3 &v) const {
    return (v[0] * axis[0]) + (v[1] * axis[1]) + (v[2] * axis[2]);
  }
};

#endif // USE_CUDA