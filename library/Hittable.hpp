#pragma once

class Ray;       // From Ray.hpp.
class Interval;  // From Interval.hpp.
class HitRecord; // From HitRecord.hpp.
class AABB;      // From AABB.hpp.

// Defines an object that can be hit by a light ray.
class Hittable {
public:
  virtual ~Hittable() = default;

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

  virtual bool bounding_box(AABB &output_box) const = 0;
};