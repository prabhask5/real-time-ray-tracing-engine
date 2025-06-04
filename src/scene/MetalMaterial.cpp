#include "MetalMaterial.hpp"
#include "../core/HitRecord.hpp"
#include "../core/Ray.hpp"
#include <Vec3Utility.hpp>

MetalMaterial::MetalMaterial(const Color &albedo, double fuzz)
    : m_albedo(albedo), m_fuzz(fuzz) {}

bool MetalMaterial::scatter(const Ray &hit_ray, const HitRecord &record,
                            Color &attenuation, Ray &scattered) const {
  // Reflect the incoming ray around the surface normal.
  Vec3 reflected_diration = reflect(hit_ray.direction(), record.normal);

  // Add some random fuzz to the direction (if fuzz > 0) and normalize to keep
  // the ray unit-length.
  reflected_diration =
      unit_vector(reflected_diration) + (m_fuzz * random_unit_vector());

  // Set the new scattered ray, and set the attenuation to the albedo of the
  // material.
  scattered = Ray(record.point, reflected_diration);
  attenuation = m_albedo;

  // Only return true if the scattered ray exits the surface (dot > 0).
  return (dot_product(scattered.direction(), record.normal) > 0);
}