#pragma once

#include "../../core/Hittable.hpp"
#include "../../core/HittableTypes.hpp"
#include "../../optimization/AABB.hpp"
#include "../../util/math/Vec3.hpp"

// Wrapper class. Rotates a hittable object around the Y-axis by a given angle.
class RotateY : public Hittable {
public:
  RotateY(HittablePtr object, double angle);

  // Getter functions.

  AABB get_bounding_box() const override;

  // Action functions.

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

private:
  // Underlying object.
  HittablePtr m_object;

  double m_sin_theta;
  double m_cos_theta;

  // New bbox after translating.
  AABB m_bbox;
};