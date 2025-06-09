#pragma once

#include "Vec3.hpp"
#include "Vec3Types.hpp"

class Ray;           // From Ray.hpp.
class HitRecord;     // From HitRecord.hpp.
class ScatterRecord; // From ScatterRecord.hpp.

// Interface for all materials, defining how they scatter incoming light rays.
class Material {
public:
  virtual ~Material() = default;

  // Action functions.

  // Used to simulate the light producing surfaces of light emitting objects.
  // Default: Color(0, 0, 0), emits black when the object does not produce
  // light. Uses u, v as the texture coordinates to control how textures emit
  // different light at different points. Point is the hit point.
  virtual Color emitted(const Ray &hit_ray, const HitRecord &record, double u,
                        double v, const Point3 &point) const {
    return Color(0, 0, 0);
  }

  // This method defines how a ray scatters (bounces, refracts, reflects) when
  // it hits a surface made of this material. NOTE: attenuation is the uutput
  // color multiplier (e.g., surface absorbs some light). and scattered is the
  // output ray (reflected or refracted). Returns false if the ray is absorbed
  // (no output ray), true if it's scattered and we can continue calculation.
  virtual bool scatter(const Ray &hit_ray, const HitRecord &record,
                       ScatterRecord &scatter_record) const {
    return false;
  }

  // Returns the probability density of scattering a ray from hit ray to
  // scattered at the given point. Needed when using importance sampling for
  // physically-based rendering.
  virtual double scattering_pdf(const Ray &hit_ray, const HitRecord &record,
                                const Ray &scattered) const {
    return 0;
  }
};
