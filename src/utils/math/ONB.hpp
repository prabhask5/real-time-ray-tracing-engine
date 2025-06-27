#pragma once

#include "SimdOps.hpp"
#include "SimdTypes.hpp"
#include "Vec3.hpp"
#include "Vec3Utility.hpp"
#include <iomanip>
#include <sstream>

// This ONB (orthonormal basis) class constructs and manages a local coordinate
// system — often used in ray tracing to convert vectors between local and world
// space, particularly for sampling directions relative to a surface normal.
// Memory layout optimized for coordinate transformations.
class alignas(16) ONB {
public:
  // Constructs an ONB from a normal vector.
  // We use n as the z-axis, and compute cross-products to find the x and y-axes
  // (they need to be perpendicular).
  ONB(const Vec3 &n) {
#if SIMD_AVAILABLE && SIMD_DOUBLE_PRECISION
    // SIMD-optimized orthonormal basis construction.

    if constexpr (SIMD_DOUBLE_PRECISION) {
      m_axis[2] = n.normalize(); // SIMD-optimized normalize
      Vec3 a = (std::fabs(m_axis[2].x()) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
      m_axis[1] =
          m_axis[2]
              .cross(a)
              .normalize(); // SIMD-optimized cross product and normalize
      m_axis[0] = m_axis[2].cross(m_axis[1]); // SIMD-optimized cross product
    } else {
#endif
      m_axis[2] = unit_vector(n);
      Vec3 a = (std::fabs(m_axis[2].x()) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
      m_axis[1] = unit_vector(cross_product(m_axis[2], a));
      m_axis[0] = cross_product(m_axis[2], m_axis[1]);
#if SIMD_AVAILABLE && SIMD_DOUBLE_PRECISION
    }
#endif
  }

  // Getter const methods.

  const Vec3 &u() const { return m_axis[0]; }
  const Vec3 &v() const { return m_axis[1]; }
  const Vec3 &w() const { return m_axis[2]; }

  // Transforms a vector from local space to world space using this ONB.
  Vec3 transform(const Vec3 &v) const {
#if SIMD_AVAILABLE && SIMD_DOUBLE_PRECISION
    // SIMD-optimized vector transformation.

    if constexpr (SIMD_DOUBLE_PRECISION) {
      Vec3 result = (v[0] * m_axis[0]); // SIMD-optimized scalar multiplication
      result +=
          (v[1] *
           m_axis[1]); // SIMD-optimized scalar multiplication and addition
      result +=
          (v[2] *
           m_axis[2]); // SIMD-optimized scalar multiplication and addition
      return result;
    }
#endif
    return (v[0] * m_axis[0]) + (v[1] * m_axis[1]) + (v[2] * m_axis[2]);
  }

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"ONB\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"u\":" << m_axis[0].json() << ",";
    oss << "\"v\":" << m_axis[1].json() << ",";
    oss << "\"w\":" << m_axis[2].json();
    oss << "}";
    return oss.str();
  }

private:
  Vec3 m_axis[3];
};
