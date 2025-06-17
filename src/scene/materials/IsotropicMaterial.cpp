#include "IsotropicMaterial.hpp"
#include "../../core/HitRecord.hpp"
#include "../../core/ScatterRecord.hpp"
#include "../../util/math/PDF.hpp"
#include "../textures/SolidColorTexture.hpp"

IsotropicMaterial::IsotropicMaterial(const Color &albedo)
    : m_texture(std::make_shared<SolidColorTexture>(albedo)) {}

IsotropicMaterial::IsotropicMaterial(TexturePtr texture) : m_texture(texture) {}

bool IsotropicMaterial::scatter(const Ray &hit_ray, const HitRecord &record,
                                ScatterRecord &scatter_record) const {
  scatter_record.attenuation =
      m_texture->value(record.u, record.v, record.point);

  // Uses a uniform sphere PDF (sphere_pdf) to scatter rays randomly in all
  // directions, regardless of normal or incident direction.
  scatter_record.pdf_ptr = std::make_shared<SpherePDF>();
  scatter_record.skip_pdf = false;

  return true;
}

double IsotropicMaterial::scattering_pdf(const Ray &hit_ray,
                                         const HitRecord &record,
                                         const Ray &scattered) const {
  // Uses a uniform sphere PDF (sphere_pdf) to scatter rays randomly in all
  // directions, regardless of normal or incident direction.
  return 1 / (4 * PI);
}