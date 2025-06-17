#include "DiffuseLightMaterial.hpp"
#include "../../core/HitRecord.hpp"
#include "../../utils/math/Vec3.hpp"
#include "../textures/SolidColorTexture.hpp"

DiffuseLightMaterial::DiffuseLightMaterial(TexturePtr texture)
    : m_texture(texture) {}

DiffuseLightMaterial::DiffuseLightMaterial(const Color &emitted_color)
    : m_texture(std::make_shared<SolidColorTexture>(emitted_color)) {}

Color DiffuseLightMaterial::emitted(const Ray &hit_ray, const HitRecord &record,
                                    double u, double v,
                                    const Point3 &point) const {
  // Only emits from the front face of the surface (checked via rec.front_face)
  // to prevent backface lighting.
  if (!record.frontFace)
    return Color(0, 0, 0);

  // Determines the color of the light emitted from the texture of the material.
  return m_texture->value(u, v, point);
}