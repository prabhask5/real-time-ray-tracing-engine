#include "AABB.hpp"
#include "../core/Ray.hpp"
#include "../utils/math/Vec3.hpp"

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

#if SIMD_AVAILABLE && SIMD_DOUBLE_PRECISION
  // SIMD-optimized ray-box intersection for x and y axes.

  if constexpr (SIMD_DOUBLE_PRECISION) {
    // Load bounding box min/max values for x and y axes.

    simd_double2 bbox_min = SimdOps::set_double2(m_x.min(), m_y.min());
    simd_double2 bbox_max = SimdOps::set_double2(m_x.max(), m_y.max());

    // Load ray origin and direction for x and y components.

    simd_double2 ray_origin = SimdOps::load_double2(origin.data());
    simd_double2 ray_dir = SimdOps::load_double2(dir.data());

    // Compute inverse direction with safety check for near-zero values.

    simd_double2 dir_inv;
    double dir_x_safe = (std::abs(dir[0]) < 1e-8) ? 1e8 : 1.0 / dir[0];
    double dir_y_safe = (std::abs(dir[1]) < 1e-8) ? 1e8 : 1.0 / dir[1];
    dir_inv = SimdOps::set_double2(dir_x_safe, dir_y_safe);

    // Compute t0 and t1 for both axes simultaneously.

    simd_double2 t0 = SimdOps::mul_double2(
        SimdOps::sub_double2(bbox_min, ray_origin), dir_inv);
    simd_double2 t1 = SimdOps::mul_double2(
        SimdOps::sub_double2(bbox_max, ray_origin), dir_inv);

    // Process x-axis.

    double t0_x = t0[0], t1_x = t1[0];
    if (t0_x > t1_x)
      std::swap(t0_x, t1_x);

    if (t0_x > t_values.min())
      t_values.set_interval(t0_x, t_values.max());
    if (t1_x < t_values.max())
      t_values.set_interval(t_values.min(), t1_x);
    if (t_values.max() <= t_values.min())
      return false;

    // Process y-axis.

    double t0_y = t0[1], t1_y = t1[1];
    if (t0_y > t1_y)
      std::swap(t0_y, t1_y);

    if (t0_y > t_values.min())
      t_values.set_interval(t0_y, t_values.max());
    if (t1_y < t_values.max())
      t_values.set_interval(t_values.min(), t1_y);
    if (t_values.max() <= t_values.min())
      return false;

    // Process z-axis separately (scalar)
    const double dir_inv_z = (std::abs(dir[2]) < 1e-8) ? 1e8 : 1.0 / dir[2];
    double t0_z = (m_z.min() - origin[2]) * dir_inv_z;
    double t1_z = (m_z.max() - origin[2]) * dir_inv_z;

    if (t0_z > t1_z)
      std::swap(t0_z, t1_z);

    if (t0_z > t_values.min())
      t_values.set_interval(t0_z, t_values.max());
    if (t1_z < t_values.max())
      t_values.set_interval(t_values.min(), t1_z);
    if (t_values.max() <= t_values.min())
      return false;

    return true;
  }
#endif

  // Fallback scalar implementation.

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