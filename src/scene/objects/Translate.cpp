#include "Translate.hpp"
#include "../../core/HitRecord.hpp"
#include "../../core/Ray.hpp"
#include "../../optimization/AABBUtility.hpp"
#include "../../utils/math/Vec3Utility.hpp"

Translate::Translate(HittablePtr object, const Vec3 &offset)
    : m_object(object), m_offset(offset) {
  m_bbox = object->get_bounding_box() + offset;
}

Translate::Translate(HittablePtr object, const Vec3 &offset, const AABB bbox)
    : m_object(object), m_offset(offset), m_bbox(bbox) {}

AABB Translate::get_bounding_box() const { return m_bbox; }

bool Translate::hit(const Ray &ray, Interval t_values,
                    HitRecord &record) const {
  // Move the ray backwards by the offset.
  Ray offset_ray(ray.origin() - m_offset, ray.direction(), ray.time());

  // Determine whether an intersection exists along the offset ray (and if so,
  // where).
  if (!m_object->hit(offset_ray, t_values, record))
    return false;

  // Move the intersection point forwards by the offset.
  record.point += m_offset;

  return true;
}

double Translate::pdf_value(const Point3 &origin, const Vec3 &direction) const {
  return m_object->pdf_value(origin - m_offset, direction);
}

Vec3 Translate::random(const Point3 &origin) const {
  return m_object->random(origin - m_offset);
}