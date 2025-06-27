#pragma once

#include "../../utils/math/SimdOps.hpp"
#include "../../utils/math/SimdTypes.hpp"
#include "Texture.hpp"
#include "TextureTypes.hpp"
#include <iomanip>
#include <sstream>

// Cache-optimized checker texture for efficient pattern evaluation.
class alignas(16) CheckerTexture : public Texture {
public:
  CheckerTexture(double scale, TexturePtr even_texture, TexturePtr odd_texture);

  CheckerTexture(double scale, const Color &c1, const Color &c2);

  Color value(double u, double v, const Point3 &p) const override;

  // Getter const methods.

  double get_scale() const;

  TexturePtr get_even_texture() const;

  TexturePtr get_odd_texture() const;

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"CheckerTexture\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"scale\":" << m_scale << ",";
    oss << "\"even_texture\":"
        << (m_even_texture ? m_even_texture->json() : "null") << ",";
    oss << "\"odd_texture\":"
        << (m_odd_texture ? m_odd_texture->json() : "null");
    oss << "}";
    return oss.str();
  }

private:
  double m_scale;
  TexturePtr m_even_texture;
  TexturePtr m_odd_texture;
};