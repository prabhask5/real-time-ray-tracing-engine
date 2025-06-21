#pragma once

#include "../../utils/math/SimdOps.hpp"
#include "../../utils/math/SimdTypes.hpp"
#include "../../utils/math/Vec3.hpp"
#include "Material.hpp"

// Models metallic (specular) surfaces like brushed aluminum or mirrors, where
// rays reflect off the surface with some optional fuzziness.
// Memory layout optimized for reflection calculations.
class alignas(16) MetalMaterial : public Material {
public:
  MetalMaterial(const Color &albedo, double fuzz);

  bool scatter(const Ray &hit_ray, const HitRecord &record,
               ScatterRecord &scatter_record) const override;

  // Getter methods for conversion.
  Color get_albedo() const;

  double get_fuzz() const;

private:
  // Defines the base color for the material.
  Color m_albedo;

  // How blurry the reflection is:
  // - fuzz = 0.0: perfect mirror.
  // - fuzz = 1.0: very blurry reflection.
  double m_fuzz;
};