#pragma once

#ifdef USE_CUDA

#include "../core/Vec3Types.hpp"
#include "math/Interval.hpp"
#include "math/Vec3.hpp"
#include <cmath>

// Applies gamma correction with gamma = 2.0.
// Converts from linear space (physical light) to gamma space (display
// brightness).
__device__ inline double linear_to_gamma(double linear_component) {
  if (linear_component > 0)
    return sqrt(linear_component);
  return 0.0;
}

// Convert a color value to an 8-bit byte after gamma correction.
__device__ inline unsigned char to_byte(double color_value) {
  double x = linear_to_gamma(color_value);

  Interval intensity(0.000, 0.999);
  return static_cast<unsigned char>(256 * intensity.clamp(x));
}

// Device-side conversion of a Color to RGB byte triplet.
__device__ inline void color_to_bytes(const Color &pixel_color,
                                      unsigned char &r, unsigned char &g,
                                      unsigned char &b) {
  r = to_byte(pixel_color.x());
  g = to_byte(pixel_color.y());
  b = to_byte(pixel_color.z());
}

#endif // USE_CUDA
