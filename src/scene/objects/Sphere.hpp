#pragma once

#include "../../core/Hittable.hpp"
#include "../../core/Ray.hpp"
#include "../../core/Vec3Types.hpp"
#include "../../optimization/AABB.hpp"
#include "../../utils/math/SimdOps.hpp"
#include "../../utils/math/SimdTypes.hpp"
#include "../../utils/math/Vec3.hpp"
#include "../materials/Material.hpp"
#include "../materials/MaterialTypes.hpp"
#include <iomanip>
#include <sstream>

// Represents a sphere hittable object.
// Memory layout optimized for ray-sphere intersection performance.
class alignas(16) Sphere : public Hittable {
public:
  Sphere(const Point3 &center, double radius, MaterialPtr material);

  Sphere(const Point3 &before_center, const Point3 &after_center, double radius,
         MaterialPtr material);

  // Initialize sphere from direct members.
  Sphere(const Ray &center, double radius, const MaterialPtr material,
         const AABB bbox);

  // Getter const methods.

  AABB get_bounding_box() const override;
  double get_radius() const { return m_radius; }
  MaterialPtr get_material() const { return m_material; }
  Ray get_center() const { return m_center; }

  // Action methods.

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

  double pdf_value(const Point3 &origin, const Vec3 &direction) const override;

  Vec3 random(const Point3 &origin) const override;

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"Sphere\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"center\":" << m_center.json() << ",";
    oss << "\"radius\":" << m_radius << ",";
    oss << "\"bbox\":" << m_bbox.json() << ",";
    oss << "\"material\":" << (m_material ? m_material->json() : "null");
    oss << "}";
    return oss.str();
  }

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