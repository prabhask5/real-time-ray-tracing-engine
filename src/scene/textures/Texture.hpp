#pragma once

#include "../../core/Vec3Types.hpp"

// Interface for all surface textures, defining the color of the material at a
// specific point.
class Texture {
public:
  virtual ~Texture() = default;

  // Used by material classes to define the surface color of the material at
  // that specific point.
  virtual Color value(double u, double v, const Point3 &p) const = 0;
};