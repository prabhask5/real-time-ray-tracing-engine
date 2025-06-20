#pragma once

#include "../optimization/AABB.hpp"
#include "Hittable.hpp"
#include "HittableTypes.hpp"
#include <vector>

// Defines a list of hittable objects.
// Memory layout optimized for ray traversal performance.
class alignas(16) HittableList : public Hittable {
public:
  HittableList();

  HittableList(HittablePtr object);

  // Getter const methods.

  AABB get_bounding_box() const override;

  std::vector<HittablePtr> &get_objects();

  const std::vector<HittablePtr> &get_objects() const;

  // Action methods.

  void clear();

  void add(HittablePtr object);

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

  double pdf_value(const Point3 &origin, const Vec3 &direction) const override;

  Vec3 random(const Point3 &origin) const override;

private:
  // Hot data: bounding box for early rejection.

  // Precomputed bounding box for the entire list.
  AABB m_bbox;

  // Warm data: object container (accessed after bbox check passes).

  // Container of hittable objects.
  std::vector<HittablePtr> m_objects;
};