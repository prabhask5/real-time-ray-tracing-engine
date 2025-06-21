#pragma once

#include "Material.hpp"

// Models transparent materials (like glass or water) that can both reflect and
// refract rays.
// Memory layout optimized for dielectric calculations.
class alignas(16) DielectricMaterial : public Material {
public:
  DielectricMaterial(double refraction_index);

  bool scatter(const Ray &hit_ray, const HitRecord &record,
               ScatterRecord &scatter_record) const override;

  // Getter method for conversion.
  double get_refraction_index() const;

private:
  // Refractive index in vacuum or air, or the ratio of the material's
  // refractive index over the refractive index of the enclosing media.
  double m_refraction_index;
};