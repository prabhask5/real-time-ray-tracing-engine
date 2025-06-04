#include "HittableList.hpp"
#include "HitRecord.hpp"
#include <Interval.hpp>

HittableList::HittableList() {}

AABB HittableList::get_bounding_box() const { return m_bbox; }

std::vector<HittablePtr> &HittableList::get_objects() { return m_objects; }

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