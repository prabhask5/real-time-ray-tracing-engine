#pragma once

#include "../../core/Hittable.hpp"
#include "../../core/HittableTypes.hpp"
#include "../../core/Vec3Types.hpp"
#include "../../utils/math/SimdOps.hpp"
#include "../../utils/math/SimdTypes.hpp"
#include "../materials/MaterialTypes.hpp"
#include "../textures/TextureTypes.hpp"

// Represents a participating medium—such as smoke, fog, or gas—that can scatter
// rays randomly inside a volume instead of just reflecting off surfaces. It
// wraps another hittable object (usually a box or sphere) and simulates light
// scattering inside it.
// Memory layout optimized for volumetric scattering calculations.
class alignas(16) ConstantMedium : public Hittable {
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

  HittablePtr get_boundary() const;

  double get_density() const;

  MaterialPtr get_phase_function() const;

private:
  // Hot data: density used in scattering probability calculations.

  double m_density; // The density of the medium

  // Warm data: boundary geometry for hit testing.

  HittablePtr m_boundary; // Container that holds the constant medium

  // Cold data: phase function accessed only on confirmed scattering.

  MaterialPtr m_phase_function; // Material defining light scattering behavior
};