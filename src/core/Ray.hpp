#pragma once

#include "../utils/math/Vec3.hpp"
#include "Vec3Types.hpp"

// Represents a light ray, represented through the parametric equation Vec3
// point = r.origin() + t * r.direction(), t is how far along the way you are (t
// is the parameter, time usually), if t = 0 we're at the origin and if t = INF
// we're infinitly far in the direction of the ray.
class Ray {
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

  // Solves the parametric equation at the parametric time value t.
  Point3 at(double t) const;

private:
  Point3 m_origin;
  Vec3 m_direction;
  double m_time;
};