#include "Translate.hpp"
#include "../../core/HitRecord.hpp"
#include "../../core/Ray.hpp"
#include "../../optimization/AABBUtility.hpp"
#include <Vec3Utility.hpp>

Translate::Translate(HittablePtr object, const Vec3 &offset)
    : m_object(object), m_offset(offset) {
  m_bbox = object->get_bounding_box() + offset;
}

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