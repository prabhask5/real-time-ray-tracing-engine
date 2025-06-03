#pragma once

#include <Hittable.hpp>
#include <MaterialTypes.hpp>
#include <Vec3.hpp>
#include <Vec3Types.hpp>

// Represents a sphere hittable object.
class Sphere : public Hittable {
public:
  Sphere(const Point3 &center, double radius, MaterialPtr material);

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

private:
  Point3 m_center;
  double m_radius;
  MaterialPtr m_material;
};