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

// Convert a color value to a byte to output to the screen.
inline unsigned char to_byte(double color_value) {
  // Apply a linear to gamma transform for gamma 2.
  double x = linear_to_gamma(color_value);

  // Translate the [0,1] component values to the byte range [0,255].
  static const Interval intensity(0.000, 0.999);
  return static_cast<unsigned char>(256 * intensity.clamp(x));
}

// Applies gamma correction, Converts float RGB values ([0, 1]) to byte range
// ([0, 255]), Writes them to a file/stream (like .ppm format).
inline void write_color(std::ostream &out, const Color &pixel_color) {
  int rbyte = to_byte(pixel_color.x());
  int gbyte = to_byte(pixel_color.y());
  int bbyte = to_byte(pixel_color.z());

  // Write out the pixel color components.
  out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}