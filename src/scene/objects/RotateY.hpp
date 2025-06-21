#pragma once

#include "../../core/Hittable.hpp"
#include "../../core/HittableTypes.hpp"
#include "../../optimization/AABB.hpp"
#include "../../utils/math/Vec3.hpp"

// Wrapper class. Rotates a hittable object around the Y-axis by a given angle.
// Memory layout optimized for rotation operations.
class alignas(16) RotateY : public Hittable {
public:
  RotateY(HittablePtr object, double angle);

  // Getter functions.

  AABB get_bounding_box() const override;

  // Action functions.

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

  double pdf_value(const Point3 &origin, const Vec3 &direction) const override;

  Vec3 random(const Point3 &origin) const override;

private:
  // Hot data: trigonometric values used in every rotation calculation.

  // Precomputed sine of rotation angle.
  double m_sin_theta;

  // Precomputed cosine of rotation angle.
  double m_cos_theta;

  // Warm data: bounding box for early rejection.

  // New bbox after rotation.
  AABB m_bbox;

  // Cold data: underlying object (accessed only after bbox checks).

  // Underlying object.
  HittablePtr m_object;
};