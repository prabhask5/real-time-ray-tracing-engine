#include "NoiseTexture.hpp"

NoiseTexture::NoiseTexture(double scale) : m_scale(scale) {}

NoiseTexture::NoiseTexture(double scale, PerlinNoise &perlin)
    : m_scale(scale), m_perlin(perlin) {}

Color NoiseTexture::value(double u, double v, const Point3 &p) const {
  // Calculates a procedural color value using sine-based noise, simulating
  // complex textures like marble. Uses a base gray color, with a sin function
  // to cause the brightness of the color to oscillate in a wavy pattern,
  // creating stripes or veins.
#if SIMD_AVAILABLE && SIMD_DOUBLE_PRECISION
  // SIMD-optimized noise texture calculation.
  if constexpr (SIMD_DOUBLE_PRECISION) {
    double turbulence =
        m_perlin.turb(p, 7); // Uses SIMD-optimized Perlin noise.
    double phase = m_scale * p.z() + 10 * turbulence;
    double factor = 1 + std::sin(phase);

    // SIMD-optimized color multiplication.
    simd_double4 base_color = SimdOps::set_double4(0.5, 0.5, 0.5, 0.0);
    simd_double4 factor_vec = SimdOps::set_double4(factor, factor, factor, 0.0);
    simd_double4 result = SimdOps::mul_double4(base_color, factor_vec);

    return Color(result[0], result[1], result[2]);
  } else {
#endif
    return Color(0.5, 0.5, 0.5) *
           (1 + std::sin(m_scale * p.z() + 10 * m_perlin.turb(p, 7)));
#if SIMD_AVAILABLE && SIMD_DOUBLE_PRECISION
  }
#endif
}

double NoiseTexture::get_scale() const { return m_scale; }

const PerlinNoise &NoiseTexture::get_perlin() const { return m_perlin; }