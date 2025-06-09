#pragma once

#include "Hittable.hpp"
#include "ONB.hpp"
#include "PDFTypes.hpp"
#include "Utility.hpp"
#include "Vec3.hpp"
#include "Vec3Utility.hpp"
#include <cmath>
#include <memory>

// Abstract base class for probability density functions.
class PDF {
public:
  virtual ~PDF() = default;

  virtual double value(const Vec3 &direction) const = 0;
  virtual Vec3 generate() const = 0;
};

// Uniform sampling over the unit sphere.
class SpherePDF : public PDF {
public:
  SpherePDF() = default;

  double value(const Vec3 &direction) const override {
    return 1.0 / (4.0 * PI);
  }

  Vec3 generate() const override { return random_unit_vector(); }
};

// Cosine-weighted hemisphere sampling PDF.
class CosinePDF : public PDF {
public:
  CosinePDF(const Vec3 &w) : m_uvw(w) {}

  double value(const Vec3 &direction) const override {
    double cosine_theta = dot_product(unit_vector(direction), m_uvw.w());
    return std::fmax(0, cosine_theta / PI);
  }

  Vec3 generate() const override {
    return m_uvw.transform(random_cosine_direction());
  }

private:
  ONB m_uvw;
};

// PDF for sampling based on a hittable object.
class HittablePDF : public PDF {
public:
  HittablePDF(const Hittable &objects, const Point3 &origin)
      : m_objects(objects), m_origin(origin) {}

  double value(const Vec3 &direction) const override {
    return m_objects.pdf_value(m_origin, direction);
  }

  Vec3 generate() const override { return m_objects.random(m_origin); }

private:
  const Hittable &m_objects;
  Point3 m_origin;
};

// Mixture of two PDFs.
class MixturePDF : public PDF {
public:
  MixturePDF(PDFPtr p0, PDFPtr p1) {
    m_p[0] = p0;
    m_p[1] = p1;
  }

  double value(const Vec3 &direction) const override {
    return 0.5 * m_p[0]->value(direction) + 0.5 * m_p[1]->value(direction);
  }

  Vec3 generate() const override {
    if (random_double() < 0.5)
      return m_p[0]->generate();
    return m_p[1]->generate();
  }

private:
  PDFPtr m_p[2];
};
