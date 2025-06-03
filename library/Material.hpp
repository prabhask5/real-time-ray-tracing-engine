#pragma once

#include "Vec3Types.hpp"

class Ray;       // From Ray.hpp.
class HitRecord; // From HitRecord.hpp.

// Interface for all materials, defining how they scatter incoming light rays.
class Material {
public:
  virtual ~Material() = default;

  // Action functions.

  // This method defines how a ray scatters (bounces, refracts, reflects) when
  // it hits a surface made of this material. NOTE: attenuation is the uutput
  // color multiplier (e.g., surface absorbs some light). and scattered is the
  // output ray (reflected or refracted). Returns false if the ray is absorbed
  // (no output ray), true if it's scattered and we can continue calculation.
  virtual bool scatter(const Ray &hit_ray, const HitRecord &record,
                       Color &attenuation, Ray &scattered) const {
    return false;
  }
};
