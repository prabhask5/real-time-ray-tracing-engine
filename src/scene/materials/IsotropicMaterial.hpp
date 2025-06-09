#pragma once

#include <Material.hpp>
#include <TextureTypes.hpp>

// Represents a scattering material where rays scatter equally in all
// directions—used for things like volumetric fog, smoke, or constant-density
// media.
class IsotropicMaterial : public Material {
public:
  IsotropicMaterial(const Color &albedo);

  IsotropicMaterial(TexturePtr texture);

  bool scatter(const Ray &hit_ray, const HitRecord &record,
               ScatterRecord &scatter_record) const override;

  double scattering_pdf(const Ray &hit_ray, const HitRecord &record,
                        const Ray &scattered) const override;

private:
  TexturePtr m_texture;
};