#pragma once

#include "../optimization/AABB.hpp"
#include <Hittable.hpp>
#include <HittableTypes.hpp>
#include <vector>

// Defines a list of hittable objects.
class HittableList : public Hittable {
public:
  HittableList();

  HittableList(HittablePtr object);

  // Getter const methods.

  AABB get_bounding_box() const override;

  std::vector<HittablePtr> &get_objects();

  // Action methods.

  void clear();

  void add(HittablePtr object);

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

  double pdf_value(const Point3 &origin, const Vec3 &direction) const override;

  Vec3 random(const Point3 &origin) const override;

private:
  std::vector<HittablePtr> m_objects;
  AABB m_bbox;
};