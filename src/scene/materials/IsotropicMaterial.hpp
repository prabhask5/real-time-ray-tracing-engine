#pragma once

#include "../textures/TextureTypes.hpp"
#include "Material.hpp"

// Represents a scattering material where rays scatter equally in all
// directions—used for things like volumetric fog, smoke, or constant-density
// media.
// Memory layout optimized for isotropic scattering calculations.
class alignas(16) IsotropicMaterial : public Material {
public:
  IsotropicMaterial(const Color &albedo);

  IsotropicMaterial(TexturePtr texture);

  bool scatter(const Ray &hit_ray, const HitRecord &record,
               ScatterRecord &scatter_record) const override;

  double scattering_pdf(const Ray &hit_ray, const HitRecord &record,
                        const Ray &scattered) const override;

  // Getter method for conversion.
  TexturePtr get_texture() const;

private:
  TexturePtr m_texture;
};