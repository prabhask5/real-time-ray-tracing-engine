#pragma once

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

private:
  std::vector<HittablePtr> m_objects;
};