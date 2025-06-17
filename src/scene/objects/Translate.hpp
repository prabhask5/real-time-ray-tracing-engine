#pragma once

#include "../../core/Hittable.hpp"
#include "../../core/HittableTypes.hpp"
#include "../../optimization/AABB.hpp"
#include "../../utils/math/Vec3.hpp"

// Wrapper class. Moves (translates) a hittable object by an offset vector in
// world space.
class Translate : public Hittable {
public:
  Translate(HittablePtr object, const Vec3 &offset);

  // Getter functions.

  AABB get_bounding_box() const override;

  // Action functions.

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

private:
  // Underlying object.
  HittablePtr m_object;

  // Vector that defines the offset distance of the object.
  Vec3 m_offset;

  // New bbox after translating.
  AABB m_bbox;
};