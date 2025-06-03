#include "LambertianMaterial.hpp"
#include "../core/HitRecord.hpp"
#include "../core/Ray.hpp"
#include <Vec3Utility.hpp>

LambertianMaterial::LambertianMaterial(const Color &albedo)
    : m_albedo(albedo) {}

bool LambertianMaterial::scatter(const Ray &hit_ray, const HitRecord &record,
                                 Color &attenuation, Ray &scattered) const {
  // Produces diffuse, hemisphere-biased scattering, simulating real-world matte
  // surfaces by adding a random unit vector to the surface normal, generating a
  // new direction roughly biased around the surface normal.
  Vec3 scatter_direction = record.normal + random_unit_vector();

  // Catch degenerate scatter direction, by reseting to normal.
  if (scatter_direction.near_zero())
    scatter_direction = record.normal;

  // Set the new scattered ray, and set the attenuation to the albedo of the
  // material.
  scattered = Ray(record.point, scatter_direction);
  attenuation = m_albedo;

  return true;
}