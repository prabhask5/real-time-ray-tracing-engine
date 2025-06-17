#pragma once

#include "Texture.hpp"
#include "TextureTypes.hpp"

class CheckerTexture : public Texture {
public:
  CheckerTexture(double scale, TexturePtr even_texture, TexturePtr odd_texture);

  CheckerTexture(double scale, const Color &c1, const Color &c2);

  Color value(double u, double v, const Point3 &p) const override;

private:
  double m_scale;
  TexturePtr m_even_texture;
  TexturePtr m_odd_texture;
};