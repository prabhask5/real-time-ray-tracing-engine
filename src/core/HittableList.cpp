#include "HittableList.hpp"
#include "HitRecord.hpp"
#include <Interval.hpp>

HittableList::HittableList() {}

HittableList::HittableList(HittablePtr object) { add(object); };

AABB HittableList::get_bounding_box() const { return m_bbox; }

std::vector<HittablePtr> &HittableList::get_objects() { return m_objects; }

void HittableList::clear() { m_objects.clear(); }

void HittableList::add(HittablePtr object) {
  m_objects.push_back(object);
  m_bbox = AABB(m_bbox, object->get_bounding_box());
}

bool HittableList::hit(const Ray &ray, Interval t_values,
                       HitRecord &record) const {
  HitRecord temp_record;
  bool hit_anything = false;
  double closest_so_far = t_values.max();

  for (const HittablePtr &object : m_objects) {
    if (object->hit(ray, Interval(t_values.min(), closest_so_far),
                    temp_record)) {
      hit_anything = true;
      closest_so_far = temp_record.t;
      record = temp_record;
    }
  }

  return hit_anything;
}

double HittableList::pdf_value(const Point3 &origin,
                               const Vec3 &direction) const {
  // For a list of hittable objects, the pdf value biasing should be just the
  // average of all the pdf value biasing of the inner hittable objects.

  if (m_objects.empty())
    return 0.0;

  double weight = 1.0 / m_objects.size();
  double sum = 0.0;

  for (const HittablePtr &object : m_objects)
    sum += weight * object->pdf_value(origin, direction);

  return sum;
}

Vec3 HittableList::random(const Point3 &origin) const {
  // Randomly chooses one object in the list and returns a direction vector
  // sampled from it.

  if (m_objects.empty())
    return Vec3(1, 0, 0);

  int int_size = int(m_objects.size());
  return m_objects[random_int(0, int_size - 1)]->random(origin);
}