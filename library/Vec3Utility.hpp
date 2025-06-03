#pragma once

#include "Vec3.hpp"
#include "Vec3Types.hpp"

inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

inline Vec3 operator*(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

inline Vec3 operator*(double t, const Vec3 &v) {
  return Vec3(t * v.x(), t * v.y(), t * v.z());
}

inline Vec3 operator*(const Vec3 &v, double t) { return t * v; }

inline Vec3 operator/(const Vec3 &v, double t) { return (1 / t) * v; }

inline double dot_product(const Vec3 &u, const Vec3 &v) {
  return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

inline Vec3 cross_product(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.y() * v.z() - u.z() * v.y(), u.z() * v.x() - u.x() * v.z(),
              u.x() * v.y() - u.y() * v.x());
}

inline Vec3 unit_vector(const Vec3 &v) { return v / v.length(); }

// Generates a random 3D point inside a unit sphere (z = 0). Used for simulating
// camera lens aperture blur (depth of field).
inline Point3 random_in_unit_disk() {
  while (true) {
    Point3 point = Point3(random_double(-1, 1), random_double(-1, 1), 0);

    // Checks whether the generated point is within the unit sphere (radius 1).
    if (point.length_squared() < 1)
      return point;
  }
}

// Generates a random unit-length vector uniformly distributed over the surface
// of the unit sphere.
inline Vec3 random_unit_vector() {
  while (true) {
    Vec3 vector = Vec3::random(-1, 1);
    double lensq = vector.length_squared();

    // Checks whether the vector is within the unit sphere.
    if (1e-160 < lensq && lensq <= 1.0)
      return vector / sqrt(lensq); // Normalizes the return vector.
  }
}

// Generates a random unit vector that lies on the same hemisphere as a given
// normal vector.
inline Vec3 random_on_hemisphere(const Vec3 &normal) {
  Vec3 on_unit_sphere = random_unit_vector();
  if (dot_product(on_unit_sphere, normal) >
      0.0) // In the same hemisphere as the normal.
    return on_unit_sphere;
  else
    return -on_unit_sphere;
}

// Computes the reflection of incident vector off a surface with normal vector.
inline Vec3 reflect(const Vec3 &incident, const Vec3 &normal) {
  return incident - 2 * dot_product(incident, normal) * normal;
}

// Computes the refraction (bending) of a light ray uv passing through a surface
// with normal n and relative refractive index etai_over_etat.
inline Vec3 refract(const Vec3 &ray, const Vec3 &normal,
                    double etai_over_etat) {
  double cos_theta = std::fmin(dot_product(-ray, normal), 1.0);
  Vec3 r_out_perp = etai_over_etat * (ray + cos_theta * normal);
  Vec3 r_out_parallel =
      -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * normal;
  return r_out_perp + r_out_parallel;
}