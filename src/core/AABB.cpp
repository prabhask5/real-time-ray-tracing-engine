#include "AABB.hpp"
#include <algorithm>

AABB::AABB() {}

AABB::AABB(const Point3 &a, const Point3 &b) : m_min(a), m_max(b) {}

const Point3 &AABB::min() const { return m_min; }

const Point3 &AABB::max() const { return m_max; }

bool AABB::hit(const Ray &r, Interval t_values) const {
  double t_min = t_values.min();
  double t_max = t_values.max();
  for (int a = 0; a < 3; a++) {
    double invD = 1.0 / r.direction()[a];
    double t0 = (m_min[a] - r.origin()[a]) * invD;
    double t1 = (m_max[a] - r.origin()[a]) * invD;
    if (invD < 0.0)
      std::swap(t0, t1);
    if (t0 > t_min)
      t_min = t0;
    if (t1 < t_max)
      t_max = t1;
    if (t_max <= t_min)
      return false;
  }
  return true;
}

AABB AABB::surrounding_box(const AABB &box0, const AABB &box1) {
  Point3 small(fmin(box0.min().x(), box1.min().x()),
               fmin(box0.min().y(), box1.min().y()),
               fmin(box0.min().z(), box1.min().z()));
  Point3 big(fmax(box0.max().x(), box1.max().x()),
             fmax(box0.max().y(), box1.max().y()),
             fmax(box0.max().z(), box1.max().z()));
  return AABB(small, big);
}
