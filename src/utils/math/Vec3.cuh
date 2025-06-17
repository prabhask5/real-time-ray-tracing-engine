#pragma once

#ifdef USE_CUDA

#include "Utility.cuh"

class Vec3 {
public:
  __device__ Vec3() : m_coordinates{0, 0, 0} {}

  __device__ Vec3(double x, double y, double z) : m_coordinates{x, y, z} {}

  // Getter methods

  __device__ double x() const { return m_coordinates[0]; }

  __device__ double y() const { return m_coordinates[1]; }

  __device__ double z() const { return m_coordinates[2]; }

  // Operator overloads for 3D Vector

  __device__ Vec3 operator-() const {
    return Vec3(-m_coordinates[0], -m_coordinates[1], -m_coordinates[2]);
  }

  __device__ double operator[](int i) const { return m_coordinates[i]; }

  __device__ double& operator[](int i) { return m_coordinates[i]; }

  __device__ Vec3& operator+=(const Vec3& v) {
    m_coordinates[0] += v.m_coordinates[0];
    m_coordinates[1] += v.m_coordinates[1];
    m_coordinates[2] += v.m_coordinates[2];
    return *this;
  }

  __device__ Vec3& operator*=(double t) {
    m_coordinates[0] *= t;
    m_coordinates[1] *= t;
    m_coordinates[2] *= t;
    return *this;
  }

  __device__ Vec3& operator/=(double t) {
    return *this *= 1.0 / t;
  }

  // Complex getter methods

  __device__ double length() const {
    return sqrt(length_squared());
  }

  __device__ double length_squared() const {
    return m_coordinates[0] * m_coordinates[0] +
           m_coordinates[1] * m_coordinates[1] +
           m_coordinates[2] * m_coordinates[2];
  }

  __device__ bool near_zero() const {
    double s = 1e-8;
    return (fabs(m_coordinates[0]) < s) &&
           (fabs(m_coordinates[1]) < s) &&
           (fabs(m_coordinates[2]) < s);
  }

  // Static methods — GPU random generation uses curandState*

  __device__ static Vec3 random(curandState* state) {
    return Vec3(random_double(state), random_double(state), random_double(state));
  }

  __device__ static Vec3 random(curandState* state, double min, double max) {
    return Vec3(random_double(state, min, max),
                random_double(state, min, max),
                random_double(state, min, max));
  }

private:
  double m_coordinates[3];
};

#endif // USE_CUDA
