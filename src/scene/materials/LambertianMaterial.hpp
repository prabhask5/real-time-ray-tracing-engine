#pragma once

#include "../../core/Vec3Types.hpp"
#include "../../utils/math/SimdOps.hpp"
#include "../../utils/math/SimdTypes.hpp"
#include "../../utils/math/Vec3.hpp"
#include "../textures/Texture.hpp"
#include "../textures/TextureTypes.hpp"
#include "Material.hpp"
#include <iomanip>
#include <sstream>

// Defines a Lambertian (diffuse) material — a surface that scatters light
// uniformly in all directions from the point of intersection.
// Cache-optimized for efficient material evaluation.
class alignas(16) LambertianMaterial : public Material {
public:
  LambertianMaterial(const Color &albedo);

  LambertianMaterial(TexturePtr texture);

  bool scatter(const Ray &hit_ray, const HitRecord &record,
               ScatterRecord &scatter_record) const override;

  double scattering_pdf(const Ray &hit_ray, const HitRecord &record,
                        const Ray &scattered) const override;

  // Getter method for conversion.
  TexturePtr get_texture() const;

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"LambertianMaterial\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"texture\":" << (m_texture ? m_texture->json() : "null");
    oss << "}";
    return oss.str();
  }

private:
  TexturePtr m_texture;
};