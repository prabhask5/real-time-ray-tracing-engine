#pragma once

#include "../../utils/math/PerlinNoise.hpp"
#include "Texture.hpp"

class NoiseTexture : public Texture {
public:
  NoiseTexture(double scale);

  Color value(double u, double v, const Point3 &p) const override;

private:
  PerlinNoise m_perlin;

  // Defines the tightness of the patterns this texture generates.
  // Higher = tighter.
  double m_scale;
};