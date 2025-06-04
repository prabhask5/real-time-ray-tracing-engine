#include "HittableList.hpp"
#include "AABB.hpp"
#include "HitRecord.hpp"
#include <Interval.hpp>

HittableList::HittableList() {}

void HittableList::clear() { m_objects.clear(); }

void HittableList::add(HittablePtr object) { m_objects.push_back(object); }

bool HittableList::hit(const Ray &ray, Interval t_values,
                       HitRecord &record) const {
  HitRecord temp_record;
  bool hit_anything = false;
  auto closest_so_far = t_values.max();

  for (const auto &object : m_objects) {
    if (object->hit(ray, Interval(t_values.min(), closest_so_far),
                    temp_record)) {
      hit_anything = true;
      closest_so_far = temp_record.t;
      record = temp_record;
    }
  }

  return hit_anything;
}

bool HittableList::bounding_box(AABB &output_box) const {
  if (m_objects.empty())
    return false;

  AABB temp_box;
  bool first_box = true;

  for (const auto &object : m_objects) {
    if (!object->bounding_box(temp_box))
      return false;
    output_box =
        first_box ? temp_box : AABB::surrounding_box(output_box, temp_box);
    first_box = false;
  }

  return true;
}

const std::vector<HittablePtr> &HittableList::objects() const {
  return m_objects;
}