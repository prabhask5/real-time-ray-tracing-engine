#pragma once

#ifdef USE_CUDA

#include "Vec3.cuh"
#include "Vec3Utility.cuh"

// This ONB (orthonormal basis) class constructs and manages a local coordinate
// system — often used in ray tracing to convert vectors between local and world
// space, particularly for sampling directions relative to a surface normal.
class ONB {
public:
  // Constructs an ONB from a normal vector.
  // We use n as the z-axis, and compute cross-products to find the x and y-axes
  // (they need to be perpendicular).
  __device__ ONB(const Vec3 &n) {
    m_axis[2] = unit_vector(n);
    Vec3 a = (fabs(m_axis[2].x()) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    m_axis[1] = unit_vector(cross_product(m_axis[2], a));
    m_axis[0] = cross_product(m_axis[2], m_axis[1]);
  }

  // Getter const methods.

  __device__ const Vec3 &u() const { return m_axis[0]; }
  __device__ const Vec3 &v() const { return m_axis[1]; }
  __device__ const Vec3 &w() const { return m_axis[2]; }

  // Transforms a vector from local space to world space using this ONB.
  __device__ Vec3 transform(const Vec3 &v) const {
    return (v[0] * m_axis[0]) + (v[1] * m_axis[1]) + (v[2] * m_axis[2]);
  }

private:
  Vec3 m_axis[3];
};

#endif // USE_CUDA
