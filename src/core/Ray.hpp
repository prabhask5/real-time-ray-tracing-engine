#pragma once

#include "../utils/math/SimdOps.hpp"
#include "../utils/math/SimdTypes.hpp"
#include "../utils/math/Vec3.hpp"
#include "Vec3Types.hpp"
#include <iomanip>
#include <sstream>

// Represents a light ray, represented through the parametric equation Vec3
// point = r.origin() + t * r.direction(), t is how far along the way you are (t
// is the parameter, time usually), if t = 0 we're at the origin and if t = INF
// we're infinitly far in the direction of the ray.
// Cache-line aligned for optimal memory access patterns.
class alignas(16) Ray {
public:
  Ray();

  Ray(const Point3 &origin, const Vec3 &direction);

  Ray(const Point3 &origin, const Vec3 &direction, double time);

  // Gets the point we're at when t = 0.
  const Point3 &origin() const;

  // Gets the vector that defines the ray's direction.
  const Vec3 &direction() const;

  // Gets the moment in time when the ray exists or is "cast". Crucial for
  // implementing motion blur and time-varying scenes in ray tracing.
  double time() const;

  // Solves the parametric equation at the parametric time value t with SIMD
  // acceleration.
  Point3 at(double t) const {
#if SIMD_AVAILABLE && SIMD_DOUBLE_PRECISION
    // SIMD-accelerated ray evaluation: origin + t * direction.
    Point3 result;

    // Load origin and direction data.
    simd_double2 origin_xy = SimdOps::load_double2(m_origin.data());
    simd_double2 direction_xy = SimdOps::load_double2(m_direction.data());

    // Scale direction by t.
    simd_double2 scaled_direction_xy =
        SimdOps::mul_scalar_double2(direction_xy, t);

    // Add to origin.
    simd_double2 result_xy =
        SimdOps::add_double2(origin_xy, scaled_direction_xy);
    SimdOps::store_double2(result.data(), result_xy);

    // Handle z component separately (since we only have double2 SIMD).
    result[2] = m_origin.z() + t * m_direction.z();

    return result;
#else
    // Fallback scalar implementation.
    return Point3(m_origin.x() + t * m_direction.x(),
                  m_origin.y() + t * m_direction.y(),
                  m_origin.z() + t * m_direction.z());
#endif
  }

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"Ray\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"origin\":" << m_origin.json() << ",";
    oss << "\"direction\":" << m_direction.json() << ",";
    oss << "\"time\":" << m_time;
    oss << "}";
    return oss.str();
  }

private:
  // Hot data: origin and direction used in every ray evaluation.

  // Ray starting point.
  Point3 m_origin;

  // Ray direction vector.
  Vec3 m_direction;

  // Cold data: time used less frequently.

  // Ray time for motion blur.
  double m_time;
};