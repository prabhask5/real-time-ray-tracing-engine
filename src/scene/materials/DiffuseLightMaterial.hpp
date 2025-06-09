#pragma once

#include <Material.hpp>
#include <TextureTypes.hpp>

// Represents a pure emissive surface—a material that emits light rather than
// reflecting or scattering it. Think of glowing surfaces like light bulbs,
// lava, or emissive panels.
class DiffuseLightMaterial : public Material {
public:
  DiffuseLightMaterial(TexturePtr texture);

  DiffuseLightMaterial(const Color &emitted_color);

  Color emitted(const Ray &hit_ray, const HitRecord &record, double u, double v,
                const Point3 &point) const override;

private:
  TexturePtr m_texture;
};