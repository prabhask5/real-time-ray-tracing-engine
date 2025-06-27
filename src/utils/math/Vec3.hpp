#pragma once

#include "SimdOps.hpp"
#include "SimdTypes.hpp"
#include "Utility.hpp"
#include <cmath>
#include <iomanip>
#include <sstream>

// SIMD-optimized Vec3 class with comprehensive ARM NEON and x86 SSE/AVX
// support. Uses double precision for mathematical accuracy with SIMD
// acceleration where possible.
class alignas(16) Vec3 {
public:
  Vec3() : m_coordinates{0, 0, 0, 0} {}

  Vec3(double x, double y, double z) : m_coordinates{x, y, z, 0} {}

  // Getter methods.

  double x() const { return m_coordinates[0]; }

  double y() const { return m_coordinates[1]; }

  double z() const { return m_coordinates[2]; }

  // Direct data access for SIMD operations.
  const double *data() const { return m_coordinates; }
  double *data() { return m_coordinates; }

  // Operator overloads for 3D Vector with SIMD acceleration.

  Vec3 operator-() const {
#if SIMD_AVAILABLE
    // Use SIMD for negation when available.
    if constexpr (SIMD_DOUBLE_PRECISION) {
      Vec3 result;
      simd_double2 xy = SimdOps::load_double2(m_coordinates);
      simd_double2 neg_xy = SimdOps::mul_scalar_double2(xy, -1.0);
      SimdOps::store_double2(result.m_coordinates, neg_xy);
      result.m_coordinates[2] = -m_coordinates[2];
      result.m_coordinates[3] = 0.0;
      return result;
    }
#endif
    return Vec3(-m_coordinates[0], -m_coordinates[1], -m_coordinates[2]);
  }

  double operator[](int i) const { return m_coordinates[i]; }

  double &operator[](int i) { return m_coordinates[i]; }

  Vec3 &operator+=(const Vec3 &v) {
#if SIMD_AVAILABLE
    // SIMD accelerated addition.
    if constexpr (SIMD_DOUBLE_PRECISION) {
      simd_double2 xy_a = SimdOps::load_double2(m_coordinates);
      simd_double2 xy_b = SimdOps::load_double2(v.m_coordinates);
      simd_double2 xy_result = SimdOps::add_double2(xy_a, xy_b);
      SimdOps::store_double2(m_coordinates, xy_result);
      m_coordinates[2] += v.m_coordinates[2];
      return *this;
    }
#endif
    m_coordinates[0] += v.m_coordinates[0];
    m_coordinates[1] += v.m_coordinates[1];
    m_coordinates[2] += v.m_coordinates[2];
    return *this;
  }

  Vec3 &operator*=(double t) {
#if SIMD_AVAILABLE
    // SIMD accelerated scalar multiplication.
    if constexpr (SIMD_DOUBLE_PRECISION) {
      simd_double2 xy = SimdOps::load_double2(m_coordinates);
      simd_double2 xy_result = SimdOps::mul_scalar_double2(xy, t);
      SimdOps::store_double2(m_coordinates, xy_result);
      m_coordinates[2] *= t;
      return *this;
    }
#endif
    m_coordinates[0] *= t;
    m_coordinates[1] *= t;
    m_coordinates[2] *= t;
    return *this;
  }

  Vec3 &operator/=(double t) { return *this *= 1.0 / t; }

  // Complex getter methods with SIMD acceleration.

  double length() const { return std::sqrt(length_squared()); }

  double length_squared() const {
#if SIMD_AVAILABLE
    // SIMD accelerated dot product with self.
    if constexpr (SIMD_DOUBLE_PRECISION) {
      simd_double2 xy = SimdOps::load_double2(m_coordinates);
      simd_double2 xy_squared = SimdOps::mul_double2(xy, xy);
      double xy_sum = xy_squared[0] + xy_squared[1];
      return xy_sum + m_coordinates[2] * m_coordinates[2];
    }
#endif
    return m_coordinates[0] * m_coordinates[0] +
           m_coordinates[1] * m_coordinates[1] +
           m_coordinates[2] * m_coordinates[2];
  }

  // Return true if the vector is close to zero in all dimensions.
  bool near_zero() const {
    constexpr double s = 1e-8;
    return (std::fabs(m_coordinates[0]) < s) &&
           (std::fabs(m_coordinates[1]) < s) &&
           (std::fabs(m_coordinates[2]) < s);
  }

  // Static methods.

  static Vec3 random() {
    return Vec3(random_double(), random_double(), random_double());
  }

  static Vec3 random(double min, double max) {
    return Vec3(random_double(min, max), random_double(min, max),
                random_double(min, max));
  }

  // SIMD-optimized utility methods.

  // Dot product with another vector.
  double dot(const Vec3 &other) const {
#if SIMD_AVAILABLE
    if constexpr (SIMD_DOUBLE_PRECISION) {
      simd_double2 xy_a = SimdOps::load_double2(m_coordinates);
      simd_double2 xy_b = SimdOps::load_double2(other.m_coordinates);
      simd_double2 xy_mul = SimdOps::mul_double2(xy_a, xy_b);
      double xy_sum = xy_mul[0] + xy_mul[1];
      return xy_sum + m_coordinates[2] * other.m_coordinates[2];
    }
#endif
    return m_coordinates[0] * other.m_coordinates[0] +
           m_coordinates[1] * other.m_coordinates[1] +
           m_coordinates[2] * other.m_coordinates[2];
  }

  // Cross product with another vector.
  Vec3 cross(const Vec3 &other) const {
    return Vec3(m_coordinates[1] * other.m_coordinates[2] -
                    m_coordinates[2] * other.m_coordinates[1],
                m_coordinates[2] * other.m_coordinates[0] -
                    m_coordinates[0] * other.m_coordinates[2],
                m_coordinates[0] * other.m_coordinates[1] -
                    m_coordinates[1] * other.m_coordinates[0]);
  }

  // Normalize this vector.
  Vec3 normalize() const {
    double len = length();
    if (len > 1e-8) {
      Vec3 result = *this;
      result *= (1.0 / len);
      return result;
    }
    return Vec3(1.0, 0.0, 0.0);
  }

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"Vec3\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"x\":" << m_coordinates[0] << ",";
    oss << "\"y\":" << m_coordinates[1] << ",";
    oss << "\"z\":" << m_coordinates[2];
    oss << "}";
    return oss.str();
  }

private:
  // 4 components for 16-byte alignment and SIMD compatibility (w component is
  // padding).
  alignas(16) double m_coordinates[4];
};