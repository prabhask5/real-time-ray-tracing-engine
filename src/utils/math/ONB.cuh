#pragma once

#ifdef USE_CUDA

#include "../../core/Vec3Types.hpp"
#include "Vec3Utility.cuh"

// Plain data structure for ONB (orthonormal basis) - no class, no methods.
// Often used in ray tracing to convert vectors between local and world space.
struct ONB {
  Vec3 axis[3]; // [0] = u, [1] = v, [2] = w (normal).
};

// Free device functions for ONB operations.
__device__ inline ONB make_onb(const Vec3 &n) {
  ONB onb;
  onb.axis[2] = unit_vector(n); // w = normal.
  Vec3 a = (fabs(onb.axis[2].x()) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
  onb.axis[1] = unit_vector(cross_product(onb.axis[2], a)); // v.
  onb.axis[0] = cross_product(onb.axis[2], onb.axis[1]);    // u.
  return onb;
}

__device__ inline Vec3 u(const ONB &onb) { return onb.axis[0]; }

__device__ inline Vec3 v(const ONB &onb) { return onb.axis[1]; }

__device__ inline Vec3 w(const ONB &onb) { return onb.axis[2]; }

__device__ inline Vec3 transform(const ONB &onb, const Vec3 &v) {
  return (v.x() * onb.axis[0]) + (v.y() * onb.axis[1]) + (v.z() * onb.axis[2]);
}

// Compatibility struct that mimics class interface for existing code.
struct ONB {
  Vec3 m_axis[3];

  __device__ ONB(const Vec3 &n) {
    m_axis[2] = unit_vector(n);
    Vec3 a = (fabs(m_axis[2].x()) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    m_axis[1] = unit_vector(cross_product(m_axis[2], a));
    m_axis[0] = cross_product(m_axis[2], m_axis[1]);
  }

  __device__ const Vec3 &u() const { return m_axis[0]; }
  __device__ const Vec3 &v() const { return m_axis[1]; }
  __device__ const Vec3 &w() const { return m_axis[2]; }

  __device__ Vec3 transform(const Vec3 &v) const {
    return (v[0] * m_axis[0]) + (v[1] * m_axis[1]) + (v[2] * m_axis[2]);
  }
};

// GPU batch processing functions (implemented in .cu file).
void batch_create_onb_from_normals(const Vec3 *d_normals, ONB *d_onbs,
                                   int count);
void batch_transform_vectors(const ONB *d_onbs, const Vec3 *d_local_vectors,
                             Vec3 *d_world_vectors, int count);

#endif // USE_CUDA
