#include "DielectricMaterial.hpp"
#include "../../core/HitRecord.hpp"
#include "../../core/Ray.hpp"
#include "../../core/ScatterRecord.hpp"
#include "../../utils/math/Vec3.hpp"
#include "../../utils/math/Vec3Utility.hpp"

DielectricMaterial::DielectricMaterial(double refraction_index)
    : m_refraction_index(refraction_index) {}

bool DielectricMaterial::scatter(const Ray &hit_ray, const HitRecord &record,
                                 ScatterRecord &scatter_record) const {
  // No absorption or color change — fully transmits light.
  scatter_record.attenuation = Color(1.0, 1.0, 1.0);

  // Set the skip pdf ray to represent a mirror reflection instead of diffuse
  // scattering.
  scatter_record.pdf_ptr = nullptr;
  scatter_record.skip_pdf = true;

  // If hitting the front face -> divide (air -> glass).
  // If hitting from inside -> use material index (glass -> air).
  double refraction_index =
      record.frontFace ? (1.0 / m_refraction_index) : m_refraction_index;

  // Use Snell's Law to determine whether total internal reflection occurs.
  // If cannot_refract == true, ray must reflect.
  Vec3 unit_direction = unit_vector(hit_ray.direction());
  double cos_theta =
      std::fmin(dot_product(-unit_direction, record.normal), 1.0);
  double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
  bool cannot_refract = refraction_index * sin_theta > 1.0;

  Vec3 direction;

  // Use Schlick's approximation for reflectance.
  double r0 = (1 - refraction_index) / (1 + refraction_index);
  r0 = r0 * r0;
  double reflectance = r0 + (1 - r0) * std::pow((1 - cos_theta), 5);

  // Use Schlick’s approximation to probabilistically reflect or refract.
  if (cannot_refract || reflectance > random_double())
    direction = reflect(unit_direction, record.normal);
  else
    direction = refract(unit_direction, record.normal, refraction_index);

  // Set the new scattered ray.
  scatter_record.skip_pdf_ray = Ray(record.point, direction);

  return true;
}