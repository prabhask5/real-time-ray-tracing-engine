#include "MetalMaterial.hpp"
#include "../../core/HitRecord.hpp"
#include "../../core/Ray.hpp"
#include "../../core/ScatterRecord.hpp"
#include "../../utils/math/Vec3Utility.hpp"

MetalMaterial::MetalMaterial(const Color &albedo, double fuzz)
    : m_albedo(albedo), m_fuzz(fuzz) {}

bool MetalMaterial::scatter(const Ray &hit_ray, const HitRecord &record,
                            ScatterRecord &scatter_record) const {
  // Reflect the incoming ray around the surface normal.
  Vec3 reflected_direction = reflect(hit_ray.direction(), record.normal);

  // Add some random fuzz to the direction (if fuzz > 0) and normalize to keep
  // the ray unit-length.
  reflected_direction =
      unit_vector(reflected_direction) + (m_fuzz * random_unit_vector());

  // Set the attenuation to the albedo of the material, and set the skip pdf ray
  // to represent a mirror reflection instead of diffuse scattering.
  scatter_record.attenuation = m_albedo;
  scatter_record.pdf_ptr = nullptr;
  scatter_record.skip_pdf = true;
  scatter_record.skip_pdf_ray =
      Ray(record.point, reflected_direction, hit_ray.time());

  return true;
}

Color MetalMaterial::get_albedo() const { return m_albedo; }

double MetalMaterial::get_fuzz() const { return m_fuzz; }