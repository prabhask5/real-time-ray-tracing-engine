#pragma once

#include "../../core/Hittable.hpp"
#include "../../optimization/AABB.hpp"
#include "../../util/math/Vec3.hpp"
#include "../materials/Material.hpp"
#include "../materials/MaterialTypes.hpp"

class Plane : public Hittable {
public:
  Plane(const Point3 &corner, const Vec3 &u_side, const Vec3 &v_side,
        MaterialPtr material);

  // Getter functions.

  AABB get_bounding_box() const override;

  // Action functions.

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

  double pdf_value(const Point3 &origin, const Vec3 &direction) const override;

  Vec3 random(const Point3 &origin) const override;

private:
  // Corner of the plane.
  Point3 m_corner;

  // Vectors spanning the rectangle's sides.
  Vec3 m_u_side, m_v_side;

  // Used to project 3D points to 2D coordinates in the quad plane.
  Vec3 m_w;

  // Surface material.
  MaterialPtr m_material;

  // Bounding box.
  AABB m_bbox;

  // Surface normal.
  Vec3 m_normal;

  // The plane equation’s constant term.
  double m_D;

  // Surface area (used in light sampling).
  double m_surface_area;
};