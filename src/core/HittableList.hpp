#pragma once

#include "AABB.hpp"
#include <Hittable.hpp>
#include <HittableTypes.hpp>
#include <vector>

// Defines a list of hittable objects.
class HittableList : public Hittable {
public:
  HittableList();

  // Action methods.

  void clear();

  void add(HittablePtr object);

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

  bool bounding_box(AABB &output_box) const override;

  const std::vector<HittablePtr> &objects() const;

private:
  std::vector<HittablePtr> m_objects;
};