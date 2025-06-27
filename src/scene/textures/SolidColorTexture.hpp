#pragma once

#include "../../utils/math/Vec3.hpp"
#include "Texture.hpp"
#include <iomanip>
#include <sstream>

// Memory layout optimized for efficient color evaluation.
class alignas(16) SolidColorTexture : public Texture {
public:
  SolidColorTexture(const Color &albedo);

  SolidColorTexture(double r, double g, double b);

  Color value(double u, double v, const Point3 &p) const override;

  // Getter const methods.

  const Color &get_albedo() const;

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"SolidColorTexture\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"albedo\":" << m_albedo.json();
    oss << "}";
    return oss.str();
  }

private:
  // Defines the base color for the material.
  Color m_albedo;
};