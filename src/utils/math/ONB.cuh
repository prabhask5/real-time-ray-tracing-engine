#pragma once

#ifdef USE_CUDA

#include "../../core/Vec3Types.cuh"
#include "Vec3Utility.cuh"

// This CudaONB (orthonormal basis) struct constructs and manages a local
// coordinate system — often used in ray tracing to convert vectors between
// local and world space, particularly for sampling directions relative to a
// surface normal.
struct CudaONB {
  CudaVec3 m_axis[3];

  // Constructs an ONB from a normal vector.
  // We use n as the z-axis, and compute cross-products to find the x and y-axes
  // (they need to be perpendicular).
  __device__ CudaONB(const CudaVec3 &n) {
    m_axis[2] = cuda_unit_vector(n);
    CudaVec3 a =
        (fabs(m_axis[2].x) > 0.9) ? CudaVec3(0, 1, 0) : CudaVec3(1, 0, 0);
    m_axis[1] = cuda_unit_vector(cuda_cross_product(m_axis[2], a));
    m_axis[0] = cuda_cross_product(m_axis[2], m_axis[1]);
  }

  // Getter const methods.

  __device__ const CudaVec3 &u() const { return m_axis[0]; }
  __device__ const CudaVec3 &v() const { return m_axis[1]; }
  __device__ const CudaVec3 &w() const { return m_axis[2]; }

  // Transforms a vector from local space to world space using this ONB.
  __device__ CudaVec3 transform(const CudaVec3 &v) const {
    return (v[0] * m_axis[0]) + (v[1] * m_axis[1]) + (v[2] * m_axis[2]);
  }
};

// GPU batch utility functions (defined in .cu file)
void cuda_batch_create_onb_from_normals(const CudaVec3 *d_normals,
                                        CudaONB *d_onbs, int count);
void cuda_batch_transform_vectors(const CudaONB *d_onbs,
                                  const CudaVec3 *d_local_vectors,
                                  CudaVec3 *d_world_vectors, int count);

#endif // USE_CUDA