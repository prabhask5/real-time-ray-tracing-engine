#include "CheckerTexture.hpp"
#include "../../utils/math/Vec3.hpp"
#include "SolidColorTexture.hpp"

CheckerTexture::CheckerTexture(double scale, TexturePtr even_texture,
                               TexturePtr odd_texture)
    : m_scale(scale), m_even_texture(even_texture), m_odd_texture(odd_texture) {
}

CheckerTexture::CheckerTexture(double scale, const Color &c1, const Color &c2)
    : CheckerTexture(scale, std::make_shared<SolidColorTexture>(c1),
                     std::make_shared<SolidColorTexture>(c2)) {}

Color CheckerTexture::value(double u, double v, const Point3 &p) const {
  double inv_scale = 1.0 / m_scale;

  // Scale the point down to the texture's scale, in all three dimensions.
  int x_index = int(std::floor(inv_scale * p.x()));
  int y_index = int(std::floor(inv_scale * p.y()));
  int z_index = int(std::floor(inv_scale * p.z()));

  // Use this to determine which checker space we're on.
  bool is_even = (x_index + y_index + z_index) % 2 == 0;

  return is_even ? m_even_texture->value(u, v, p)
                 : m_odd_texture->value(u, v, p);
}

double CheckerTexture::get_scale() const { return m_scale; }

TexturePtr CheckerTexture::get_even_texture() const { return m_even_texture; }

TexturePtr CheckerTexture::get_odd_texture() const { return m_odd_texture; }