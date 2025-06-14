#pragma once

#include "../../core/Ray.hpp"
#include "../../optimization/AABB.hpp"
#include <Hittable.hpp>
#include <MaterialTypes.hpp>
#include <Vec3.hpp>
#include <Vec3Types.hpp>

// Represents a sphere hittable object.
class Sphere : public Hittable {
public:
  Sphere(const Point3 &center, double radius, MaterialPtr material);

  Sphere(const Point3 &before_center, const Point3 &after_center, double radius,
         MaterialPtr material);

  // Getter const methods.

  AABB get_bounding_box() const override;

  // Action methods.

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

  double pdf_value(const Point3 &origin, const Vec3 &direction) const override;

  Vec3 random(const Point3 &origin) const override;

private:
  Ray m_center;
  double m_radius;
  MaterialPtr m_material;
  AABB m_bbox;
};