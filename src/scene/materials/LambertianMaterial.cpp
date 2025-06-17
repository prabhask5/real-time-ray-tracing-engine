#include "LambertianMaterial.hpp"
#include "../../core/HitRecord.hpp"
#include "../../core/Ray.hpp"
#include "../../core/ScatterRecord.hpp"
#include "../../utils/math/PDF.hpp"
#include "../../utils/math/Vec3Utility.hpp"
#include "../textures/SolidColorTexture.hpp"

LambertianMaterial::LambertianMaterial(const Color &albedo)
    : m_texture(std::make_shared<SolidColorTexture>(albedo)) {}

LambertianMaterial::LambertianMaterial(TexturePtr texture)
    : m_texture(texture) {}

bool LambertianMaterial::scatter(const Ray &hit_ray, const HitRecord &record,
                                 ScatterRecord &scatter_record) const {
  // Produces diffuse, hemisphere-biased scattering, simulating real-world matte
  // surfaces by adding a random unit vector to the surface normal, generating a
  // new direction roughly biased around the surface normal.

  // Set the attenuation to the albedo of the
  // material.
  scatter_record.attenuation =
      m_texture->value(record.u, record.v, record.point);

  // This assigns a cosine-weighted probability density function (PDF) for
  // generating outgoing rays. This is ideal for diffuse materials, where light
  // scatters in many directions but is stronger near the normal.
  scatter_record.pdf_ptr = std::make_shared<CosinePDF>(record.normal);
  scatter_record.skip_pdf = false;

  return true;
}

double LambertianMaterial::scattering_pdf(const Ray &hit_ray,
                                          const HitRecord &record,
                                          const Ray &scattered) const {
  double cos_theta =
      dot_product(record.normal, unit_vector(scattered.direction()));

  // PDF Formula for Lambertian materials.
  return cos_theta < 0 ? 0 : cos_theta / PI;
}