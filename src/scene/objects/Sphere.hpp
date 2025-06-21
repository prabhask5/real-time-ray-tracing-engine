#pragma once

#include "../../core/Hittable.hpp"
#include "../../core/Ray.hpp"
#include "../../core/Vec3Types.hpp"
#include "../../optimization/AABB.hpp"
#include "../../utils/math/Vec3.hpp"
#include "../materials/MaterialTypes.hpp"

// Represents a sphere hittable object.
// Memory layout optimized for ray-sphere intersection performance.
class alignas(16) Sphere : public Hittable {
public:
  Sphere(const Point3 &center, double radius, MaterialPtr material);

  Sphere(const Point3 &before_center, const Point3 &after_center, double radius,
         MaterialPtr material);

  // Getter const methods.

  AABB get_bounding_box() const override;
  Point3 get_center() const {
    return m_center.at(0.5);
  } // Get center at time 0.5
  Point3 get_center(double time) const { return m_center.at(time); }
  double get_radius() const { return m_radius; }
  MaterialPtr get_material() const { return m_material; }
  Ray get_center_ray() const { return m_center; }

  // Action methods.

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

  double pdf_value(const Point3 &origin, const Vec3 &direction) const override;

  Vec3 random(const Point3 &origin) const override;

private:
  // Hot data: most frequently accessed in ray-sphere intersection.

  // Center position (motion blur support).
  Ray m_center;

  // Sphere radius - grouped with center for intersection math.
  double m_radius;

  // Warm data: used for early rejection.

  // Bounding box for acceleration structure.
  AABB m_bbox;

  // Cold data: accessed only on confirmed hits.

  // Material properties.
  MaterialPtr m_material;
};