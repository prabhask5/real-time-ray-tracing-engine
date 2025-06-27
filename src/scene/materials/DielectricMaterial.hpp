#pragma once

#include "../../utils/math/SimdOps.hpp"
#include "../../utils/math/SimdTypes.hpp"
#include "Material.hpp"
#include <iomanip>
#include <sstream>

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

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"DielectricMaterial\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"refraction_index\":" << m_refraction_index;
    oss << "}";
    return oss.str();
  }

private:
  // Refractive index in vacuum or air, or the ratio of the material's
  // refractive index over the refractive index of the enclosing media.
  double m_refraction_index;
};