#pragma once

#include "Ray.hpp"
#include <Interval.hpp>
#include <Vec3.hpp>
#include <Vec3Types.hpp>

class AABB {
public:
  AABB();
  AABB(const Point3 &a, const Point3 &b);

  const Point3 &min() const;
  const Point3 &max() const;

  bool hit(const Ray &r, Interval t_values) const;

  static AABB surrounding_box(const AABB &box0, const AABB &box1);

private:
  Point3 m_min;
  Point3 m_max;
};
