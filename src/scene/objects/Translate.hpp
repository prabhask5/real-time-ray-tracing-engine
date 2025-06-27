#pragma once

#include "../../core/Hittable.hpp"
#include "../../core/HittableTypes.hpp"
#include "../../optimization/AABB.hpp"
#include "../../utils/math/Vec3.hpp"
#include <iomanip>
#include <sstream>

// Wrapper class. Moves (translates) a hittable object by an offset vector in
// world space.
// Memory layout optimized for translation operations.
class alignas(16) Translate : public Hittable {
public:
  Translate(HittablePtr object, const Vec3 &offset);

  // Initialize translate from direct members.
  Translate(HittablePtr object, const Vec3 &offset, const AABB bbox);

  // Getter functions.

  AABB get_bounding_box() const override;

  HittablePtr get_object() const { return m_object; }

  Vec3 get_offset() const { return m_offset; }

  // Action functions.

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

  double pdf_value(const Point3 &origin, const Vec3 &direction) const override;

  Vec3 random(const Point3 &origin) const override;

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"Translate\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"offset\":" << m_offset.json() << ",";
    oss << "\"bbox\":" << m_bbox.json() << ",";
    oss << "\"object\":" << (m_object ? m_object->json() : "null");
    oss << "}";
    return oss.str();
  }

private:
  // Hot data: translation offset used in every transform.

  // Vector that defines the offset distance.
  Vec3 m_offset;

  // Warm data: bounding box for early rejection.

  // New bbox after translating.
  AABB m_bbox;

  // Cold data: underlying object (accessed only after bbox/offset checks).

  // Underlying object.
  HittablePtr m_object;
};