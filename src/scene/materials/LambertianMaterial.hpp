#pragma once

#include <Material.hpp>
#include <Vec3.hpp>
#include <Vec3Types.hpp>

// Defines a Lambertian (diffuse) material — a surface that scatters light
// uniformly in all directions from the point of intersection.
class LambertianMaterial : public Material {
public:
  LambertianMaterial(const Color &albedo);

  bool scatter(const Ray &hit_ray, const HitRecord &record, Color &attenuation,
               Ray &scattered) const override;

private:
  // Defines the base color for the material.
  Color m_albedo;
};