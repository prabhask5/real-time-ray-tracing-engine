#pragma once

#include "../../core/Hittable.hpp"
#include "../../optimization/AABB.hpp"
#include "../../utils/math/SimdOps.hpp"
#include "../../utils/math/SimdTypes.hpp"
#include "../../utils/math/Vec3.hpp"
#include "../materials/Material.hpp"
#include "../materials/MaterialTypes.hpp"
#include <iomanip>
#include <sstream>

// Memory layout optimized for ray-plane intersection performance.
class alignas(16) Plane : public Hittable {
public:
  Plane(const Point3 &corner, const Vec3 &u_side, const Vec3 &v_side,
        MaterialPtr material);

  // Getter functions.

  AABB get_bounding_box() const override;
  Point3 get_corner() const { return m_corner; }
  Vec3 get_u_side() const { return m_u_side; }
  Vec3 get_v_side() const { return m_v_side; }
  Vec3 get_normal() const { return m_normal; }
  MaterialPtr get_material() const { return m_material; }
  double get_surface_area() const { return m_surface_area; }

  // Action functions.

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

  double pdf_value(const Point3 &origin, const Vec3 &direction) const override;

  Vec3 random(const Point3 &origin) const override;

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"Plane\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"corner\":" << m_corner.json() << ",";
    oss << "\"u_side\":" << m_u_side.json() << ",";
    oss << "\"v_side\":" << m_v_side.json() << ",";
    oss << "\"normal\":" << m_normal.json() << ",";
    oss << "\"w\":" << m_w.json() << ",";
    oss << "\"d\":" << m_D << ",";
    oss << "\"surface_area\":" << m_surface_area << ",";
    oss << "\"material\":" << (m_material ? m_material->json() : "null");
    oss << "}";
    return oss.str();
  }

private:
  // Hot data: core geometry accessed in every ray-plane intersection.

  // Corner of the plane.
  Point3 m_corner;

  // Vectors spanning the rectangle's sides.
  Vec3 m_u_side, m_v_side;

  // Surface normal.
  Vec3 m_normal;

  // Used to project 3D points to 2D coordinates.
  Vec3 m_w;

  // The plane equation’s constant term.
  double m_D;

  // Surface area (used in light sampling).
  double m_surface_area;

  // Warm data: early rejection and PDF calculations.

  // Bounding box.
  AABB m_bbox;

  // Cold data: accessed only on confirmed hits.

  // Surface material.
  MaterialPtr m_material;
};