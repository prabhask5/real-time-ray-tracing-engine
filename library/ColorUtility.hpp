#pragma once

#include "Interval.hpp"
#include "Vec3.hpp"
#include "Vec3Types.hpp"
#include <iostream>

// Applies gamma correction with gamma = 2.0. This converts from linear color
// space (how light physically behaves) to gamma-encoded space (how displays
// represent brightness).
inline double linear_to_gamma(double linear_component) {
  if (linear_component > 0)
    return std::sqrt(linear_component);

  return 0;
}

// Applies gamma correction, Converts float RGB values ([0, 1]) to byte range
// ([0, 255]), Writes them to a file/stream (like .ppm format).
void write_color(std::ostream &out, const Color &pixel_color) {
  double r = pixel_color.x();
  double g = pixel_color.y();
  double b = pixel_color.z();

  // Apply a linear to gamma transform for gamma 2.
  r = linear_to_gamma(r);
  g = linear_to_gamma(g);
  b = linear_to_gamma(b);

  // Translate the [0,1] component values to the byte range [0,255].
  static const Interval intensity(0.000, 0.999);
  int rbyte = int(256 * intensity.clamp(r));
  int gbyte = int(256 * intensity.clamp(g));
  int bbyte = int(256 * intensity.clamp(b));

  // Write out the pixel color components.
  out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}