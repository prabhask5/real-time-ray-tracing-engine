#pragma once

#include "../../utils/math/Vec3.hpp"
#include "Texture.hpp"

class SolidColorTexture : public Texture {
public:
  SolidColorTexture(const Color &albedo);

  SolidColorTexture(double r, double g, double b);

  Color value(double u, double v, const Point3 &p) const override;

  // Getter const methods.

  const Color &get_albedo() const;

private:
  // Defines the base color for the material.
  Color m_albedo;
};