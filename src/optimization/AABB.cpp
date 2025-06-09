#include "AABB.hpp"
#include "../core/Ray.hpp"
#include <Vec3.hpp>

AABB::AABB() {}

AABB::AABB(const Interval &x, const Interval &y, const Interval &z)
    : m_x(x), m_y(y), m_z(z) {
  pad_to_minimums();
}

AABB::AABB(const Point3 &p1, const Point3 &p2) {
  m_x =
      (p1.x() <= p2.x()) ? Interval(p1.x(), p2.x()) : Interval(p2.x(), p1.x());
  m_y =
      (p1.y() <= p2.y()) ? Interval(p1.y(), p2.y()) : Interval(p2.y(), p1.y());
  m_z =
      (p1.z() <= p2.z()) ? Interval(p1.z(), p2.z()) : Interval(p2.z(), p1.z());

  pad_to_minimums();
}

AABB::AABB(const AABB &b1, const AABB &b2) {
  m_x = Interval(b1.x(), b2.x());
  m_y = Interval(b1.y(), b2.y());
  m_z = Interval(b1.z(), b2.z());
}

const Interval &AABB::x() const { return m_x; }

const Interval &AABB::y() const { return m_y; }

const Interval &AABB::z() const { return m_z; }

const Interval &AABB::get_axis_interval(int index) const {
  if (index == 1)
    return m_y;
  if (index == 2)
    return m_z;
  return m_x;
}

int AABB::get_longest_axis() const {
  if (m_x.size() > m_y.size())
    return m_x.size() > m_z.size() ? 0 : 2;
  else
    return m_y.size() > m_z.size() ? 1 : 2;
}

bool AABB::hit(const Ray &ray, Interval t_values) const {
  const Point3 &origin = ray.origin();
  const Vec3 &dir = ray.direction();

  for (int axis = 0; axis < 3; axis++) {
    const Interval &ax = get_axis_interval(axis);
    const double dir_inv = 1.0 / dir[axis];

    double t0 = (ax.min() - origin[axis]) * dir_inv;
    double t1 = (ax.max() - origin[axis]) * dir_inv;

    if (t0 < t1) {
      if (t0 > t_values.min())
        t_values.set_interval(t0, t_values.max());
      if (t1 < t_values.max())
        t_values.set_interval(t_values.min(), t1);
    } else {
      if (t1 > t_values.min())
        t_values.set_interval(t1, t_values.max());
      if (t0 < t_values.max())
        t_values.set_interval(t_values.min(), t0);
    }

    if (t_values.max() <= t_values.min())
      return false;
  }

  return true;
}

void AABB::pad_to_minimums() {
  double delta = 0.0001;

  if (m_x.size() < delta)
    m_x = m_x.expand(delta);
  if (m_y.size() < delta)
    m_y = m_y.expand(delta);
  if (m_z.size() < delta)
    m_z = m_z.expand(delta);
}