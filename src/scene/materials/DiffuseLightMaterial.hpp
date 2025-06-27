#pragma once

#include "../textures/Texture.hpp"
#include "../textures/TextureTypes.hpp"
#include "Material.hpp"
#include <iomanip>
#include <sstream>

// Represents a pure emissive surface—a material that emits light rather than
// reflecting or scattering it. Think of glowing surfaces like light bulbs,
// lava, or emissive panels.
// Memory layout optimized for light emission calculations.
class alignas(16) DiffuseLightMaterial : public Material {
public:
  DiffuseLightMaterial(TexturePtr texture);

  DiffuseLightMaterial(const Color &emitted_color);

  Color emitted(const Ray &hit_ray, const HitRecord &record, double u, double v,
                const Point3 &point) const override;

  // Getter method for conversion.
  TexturePtr get_texture() const;

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"DiffuseLightMaterial\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"texture\":" << (m_texture ? m_texture->json() : "null");
    oss << "}";
    return oss.str();
  }

private:
  TexturePtr m_texture;
};