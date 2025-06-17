#pragma once

#include "../util/math/Vec3.hpp"
#include "Vec3Types.hpp"

class Ray;       // From Ray.hpp.
class Interval;  // From Interval.hpp.
class HitRecord; // From HitRecord.hpp.
class AABB;      // From AABB.hpp.

// Defines an object that can be hit by a light ray.
class Hittable {
public:
  virtual ~Hittable() = default;

  // Getter functions.

  virtual AABB get_bounding_box() const = 0;

  // Action functions.

  // This function checks if the ray hits the hittable object with the t values
  // in the interval range ray_t- if true, writes the hit record info in the hit
  // record reference object. NOTE: A ray can be represented through the
  // parametric equation Vec3 point = r.origin() + t * r.direction(), t is how
  // far along the way you are (t is the parameter, time usually), if t = 0
  // we're at the origin and if t = INF we're infinitly far in the direction of
  // the ray. This function takes in an interval on t values to search through
  // to determine if the ray intersected with the hittable during that time.
  virtual bool hit(const Ray &ray, Interval t_values,
                   HitRecord &record) const = 0;

  // Returns the probability density function (PDF) value for shooting a ray
  // from origin in the given direction toward this object. To do smart light
  // sampling (useful when adding light sources to an environment) we need to
  // bias how much sampling we take from a particular direction. This PDF value
  // is used to weight the color contribution from that direction properly.
  virtual double pdf_value(const Point3 &origin, const Vec3 &direction) const {
    return 0.0;
  }

  // Generates a random direction vector from the origin that aims toward the
  // object, according to the object's own distribution. We let each hittable
  // object define this to change how random sampling works for each type of
  // object.
  virtual Vec3 random(const Point3 &origin) const { return Vec3(1, 0, 0); }
};