#pragma once

#include "../../core/Hittable.hpp"
#include "../../optimization/AABB.hpp"
#include "../../utils/math/Vec3.hpp"
#include "../materials/Material.hpp"
#include "../materials/MaterialTypes.hpp"

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

private:
  // Hot data: core geometry accessed in every ray-plane intersection
  Point3 m_corner; // Corner of the plane

  Vec3 m_u_side, m_v_side; // Vectors spanning the rectangle's sides
  Vec3 m_normal;           // Surface normal
  Vec3 m_w;                // Used to project 3D points to 2D coordinates

  // The plane equation’s constant term.
  double m_D;

  // Surface area (used in light sampling).
  double m_surface_area;

  // Warm data: early rejection and PDF calculations
  AABB m_bbox; // Bounding box

  // Cold data: accessed only on confirmed hits
  MaterialPtr m_material; // Surface material
};