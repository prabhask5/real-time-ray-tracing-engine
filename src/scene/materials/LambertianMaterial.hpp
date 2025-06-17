#pragma once

#include "../../core/Vec3Types.hpp"
#include "../../utils/math/Vec3.hpp"
#include "../textures/TextureTypes.hpp"
#include "Material.hpp"

// Defines a Lambertian (diffuse) material — a surface that scatters light
// uniformly in all directions from the point of intersection.
class LambertianMaterial : public Material {
public:
  LambertianMaterial(const Color &albedo);

  LambertianMaterial(TexturePtr texture);

  bool scatter(const Ray &hit_ray, const HitRecord &record,
               ScatterRecord &scatter_record) const override;

  double scattering_pdf(const Ray &hit_ray, const HitRecord &record,
                        const Ray &scattered) const override;

private:
  TexturePtr m_texture;
};