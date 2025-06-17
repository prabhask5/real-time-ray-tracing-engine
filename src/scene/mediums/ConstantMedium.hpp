#pragma once

#include "../../core/Hittable.hpp"
#include "../../core/HittableTypes.hpp"
#include "../../core/Vec3Types.hpp"
#include "../materials/MaterialTypes.hpp"
#include "../textures/TextureTypes.hpp"

// Represents a participating medium—such as smoke, fog, or gas—that can scatter
// rays randomly inside a volume instead of just reflecting off surfaces. It
// wraps another hittable object (usually a box or sphere) and simulates light
// scattering inside it.
class ConstantMedium : public Hittable {
public:
  ConstantMedium(HittablePtr boundary, double density, TexturePtr texture);

  ConstantMedium(HittablePtr boundary, double density, const Color &albedo);

  // Getter functions.

  AABB get_bounding_box() const override;

  // Action functions.

  // Handles ray interaction with volumetric media—like fog, smoke, or clouds—by
  // determining whether a ray randomly scatters inside the medium instead of
  // just passing through it. It's used in ray tracing to simulate participating
  // media.
  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

private:
  // Defines the container that holds the constant medium (smoke or fog).
  HittablePtr m_boundary;

  // The density of the medium.
  double m_density;

  // The material that defines how light is scattered inside of the medium.
  MaterialPtr m_phase_function;
};