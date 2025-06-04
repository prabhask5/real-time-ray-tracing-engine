#include "SolidColorTexture.hpp"

SolidColorTexture::SolidColorTexture(const Color &albedo) : m_albedo(albedo) {}

SolidColorTexture::SolidColorTexture(double r, double g, double b)
    : SolidColorTexture(Color(r, g, b)) {}

Color SolidColorTexture::value(double u, double v, const Point3 &p) const {
  return m_albedo;
}