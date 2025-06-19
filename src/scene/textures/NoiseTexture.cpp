#include "NoiseTexture.hpp"

NoiseTexture::NoiseTexture(double scale) : m_scale(scale) {}

NoiseTexture::NoiseTexture(double scale, PerlinNoise &perlin)
    : m_scale(scale), m_perlin(perlin) {}

Color NoiseTexture::value(double u, double v, const Point3 &p) const {
  // Calculates a procedural color value using sine-based noise, simulating
  // complex textures like marble. Uses a base gray color, with a sin function
  // to cause the brightness of the color to oscillate in a wavy pattern,
  // creating stripes or veins.
  return Color(0.5, 0.5, 0.5) *
         (1 + std::sin(m_scale * p.z() + 10 * m_perlin.turb(p, 7)));
}

double NoiseTexture::get_scale() const { return m_scale; }

const PerlinNoise &NoiseTexture::get_perlin() const { return m_perlin; }