#pragma once

#include "../../utils/math/PerlinNoise.hpp"
#include "../../utils/math/SimdOps.hpp"
#include "../../utils/math/SimdTypes.hpp"
#include "Texture.hpp"
#include <iomanip>
#include <sstream>

// Memory layout optimized for noise pattern generation.
class alignas(16) NoiseTexture : public Texture {
public:
  NoiseTexture(double scale);

  NoiseTexture(double scale, PerlinNoise &perlin);

  Color value(double u, double v, const Point3 &p) const override;

  // Getter const methods.

  double get_scale() const;

  const PerlinNoise &get_perlin() const;

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"NoiseTexture\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"scale\":" << m_scale << ",";
    oss << "\"perlin\":" << m_perlin.json();
    oss << "}";
    return oss.str();
  }

private:
  // Hot data: frequently accessed scale parameter.

  // Defines the tightness of the patterns (higher = tighter).
  double m_scale;

  // Warm data: noise generator.

  // Perlin noise generator.
  PerlinNoise m_perlin;
};