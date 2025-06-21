#pragma once

#include "Utility.hpp"

// SIMD-optimized Vec3 class with 16-byte alignment for future SSE/AVX
// integration.
class alignas(16) Vec3 {
public:
  Vec3() : m_coordinates{0, 0, 0, 0} {}

  Vec3(double x, double y, double z) : m_coordinates{x, y, z, 0} {}

  // Getter methods const methods.

  double x() const { return m_coordinates[0]; }

  double y() const { return m_coordinates[1]; }

  double z() const { return m_coordinates[2]; }

  // Operator overloads for 3D Vector.

  Vec3 operator-() const {
    return Vec3(-m_coordinates[0], -m_coordinates[1], -m_coordinates[2]);
  }

  double operator[](int i) const { return m_coordinates[i]; }

  double &operator[](int i) { return m_coordinates[i]; }

  Vec3 &operator+=(const Vec3 &v) {
    m_coordinates[0] += v.m_coordinates[0];
    m_coordinates[1] += v.m_coordinates[1];
    m_coordinates[2] += v.m_coordinates[2];

    return *this;
  }

  Vec3 &operator*=(double t) {
    m_coordinates[0] *= t;
    m_coordinates[1] *= t;
    m_coordinates[2] *= t;
    return *this;
  }

  Vec3 &operator/=(double t) { return *this *= 1 / t; }

  // Complex getter methods.

  double length() const { return std::sqrt(length_squared()); }

  double length_squared() const {
    return m_coordinates[0] * m_coordinates[0] +
           m_coordinates[1] * m_coordinates[1] +
           m_coordinates[2] * m_coordinates[2];
  }

  // Return true if the vector is close to zero in all dimensions.
  bool near_zero() const {
    double s = 1e-8;
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

private:
  // 4 components for 16-byte alignment and SIMD compatibility (w component is
  // padding).
  double m_coordinates[4];
};