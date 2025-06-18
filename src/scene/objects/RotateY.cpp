#include "RotateY.hpp"
#include "../../core/HitRecord.hpp"
#include "../../core/Ray.hpp"

RotateY::RotateY(HittablePtr object, double angle) : m_object(object) {
  double radians = degrees_to_radians(angle);
  m_sin_theta = std::sin(radians);
  m_cos_theta = std::cos(radians);
  m_bbox = object->get_bounding_box();

  Point3 min(INF, INF, INF);
  Point3 max(-INF, -INF, -INF);

  // Computes the diagonal 3D space points of the rotated bbox by using
  // precomputed sin and cos theta values.
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        double x = i * m_bbox.x().max() + (1 - i) * m_bbox.x().min();
        double y = j * m_bbox.y().max() + (1 - j) * m_bbox.y().min();
        double z = k * m_bbox.z().max() + (1 - k) * m_bbox.z().min();

        double new_x = m_cos_theta * x + m_sin_theta * z;
        double new_z = -m_sin_theta * x + m_cos_theta * z;

        Vec3 tester(new_x, y, new_z);

        for (int c = 0; c < 3; c++) {
          min[c] = std::fmin(min[c], tester[c]);
          max[c] = std::fmax(max[c], tester[c]);
        }
      }
    }
  }

  m_bbox = AABB(min, max);
}

AABB RotateY::get_bounding_box() const { return m_bbox; }

bool RotateY::hit(const Ray &ray, Interval t_values, HitRecord &record) const {
  // Transform the ray from world space to object space.

  Point3 origin = Point3(
      (m_cos_theta * ray.origin().x()) - (m_sin_theta * ray.origin().z()),
      ray.origin().y(),
      (m_sin_theta * ray.origin().x()) + (m_cos_theta * ray.origin().z()));

  Vec3 direction = Vec3((m_cos_theta * ray.direction().x()) -
                            (m_sin_theta * ray.direction().z()),
                        ray.direction().y(),
                        (m_sin_theta * ray.direction().x()) +
                            (m_cos_theta * ray.direction().z()));

  Ray rotated_ray(origin, direction, ray.time());

  // Determine whether an intersection exists in object space (and if so,
  // where).

  if (!m_object->hit(rotated_ray, t_values, record))
    return false;

  // Transform the intersection from object space back to world space.

  record.point = Point3(
      (m_cos_theta * record.point.x()) + (m_sin_theta * record.point.z()),
      record.point.y(),
      (-m_sin_theta * record.point.x()) + (m_cos_theta * record.point.z()));

  record.normal = Vec3(
      (m_cos_theta * record.normal.x()) + (m_sin_theta * record.normal.z()),
      record.normal.y(),
      (-m_sin_theta * record.normal.x()) + (m_cos_theta * record.normal.z()));

  return true;
}

double RotateY::pdf_value(const Point3 &origin, const Vec3 &direction) const {
  // Transform origin and direction to object space.
  Point3 rotated_origin(m_cos_theta * origin.x() - m_sin_theta * origin.z(),
                        origin.y(),
                        m_sin_theta * origin.x() + m_cos_theta * origin.z());

  Vec3 rotated_direction(
      m_cos_theta * direction.x() - m_sin_theta * direction.z(), direction.y(),
      m_sin_theta * direction.x() + m_cos_theta * direction.z());

  return m_object->pdf_value(rotated_origin, rotated_direction);
}

Vec3 RotateY::random(const Point3 &origin) const {
  // Transform origin to object space.
  Point3 rotated_origin(m_cos_theta * origin.x() - m_sin_theta * origin.z(),
                        origin.y(),
                        m_sin_theta * origin.x() + m_cos_theta * origin.z());

  // Get random direction in object space.
  Vec3 obj_dir = m_object->random(rotated_origin);

  // Transform back to world space.
  return Vec3(m_cos_theta * obj_dir.x() + m_sin_theta * obj_dir.z(),
              obj_dir.y(),
              -m_sin_theta * obj_dir.x() + m_cos_theta * obj_dir.z());
}