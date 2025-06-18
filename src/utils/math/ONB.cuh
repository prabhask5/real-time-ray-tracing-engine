#pragma once

#ifdef USE_CUDA

#include "../../core/Vec3Types.cuh"
#include "Vec3Utility.cuh"

// POD struct for GPU orthonormal basis (ONB)
struct CudaONB {
  CudaVec3 axis[3]; // [0] = u, [1] = v, [2] = w (normal).
};

// Constructs an ONB from a normal vector.
__device__ inline CudaONB cuda_onb_from_normal(const CudaVec3 &n) {
  CudaONB onb;
  onb.axis[2] = cuda_unit_vector(n); // w-axis
  CudaVec3 a =
      (fabs(onb.axis[2].x()) > 0.9) ? CudaVec3(0, 1, 0) : CudaVec3(1, 0, 0);
  onb.axis[1] = cuda_unit_vector(cuda_cross_product(onb.axis[2], a)); // v-axis
  onb.axis[0] = cuda_cross_product(onb.axis[2], onb.axis[1]);         // u-axis
  return onb;
}

// Component accessors
__device__ inline CudaVec3 cuda_onb_u(const CudaONB &onb) {
  return onb.axis[0];
}
__device__ inline CudaVec3 cuda_onb_v(const CudaONB &onb) {
  return onb.axis[1];
}
__device__ inline CudaVec3 cuda_onb_w(const CudaONB &onb) {
  return onb.axis[2];
}

// Transforms a vector from local to world space using the ONB
__device__ inline CudaVec3 cuda_onb_transform(const CudaONB &onb,
                                              const CudaVec3 &v) {
  return (v.x() * onb.axis[0]) + (v.y() * onb.axis[1]) + (v.z() * onb.axis[2]);
}

// GPU batch utility functions (defined in .cu file)
void cuda_batch_create_onb_from_normals(const CudaVec3 *d_normals,
                                        CudaONB *d_onbs, int count);

void cuda_batch_transform_vectors(const CudaONB *d_onbs,
                                  const CudaVec3 *d_local_vectors,
                                  CudaVec3 *d_world_vectors, int count);

#endif // USE_CUDA
