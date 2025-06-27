#pragma once

#include "../../core/Hittable.hpp"
#include "ONB.hpp"
#include "PDFTypes.hpp"
#include "Utility.hpp"
#include "Vec3.hpp"
#include "Vec3Utility.hpp"
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>

// Abstract base class for probability density functions.
// Memory layout optimized for PDF calculations.
class alignas(16) PDF {
public:
  virtual ~PDF() = default;

  virtual double value(const Vec3 &direction) const = 0;
  virtual Vec3 generate() const = 0;

  // JSON serialization method.
  virtual std::string json() const = 0;
};

// Uniform sampling over the unit sphere.
// Optimized for uniform sphere sampling.
class alignas(16) SpherePDF : public PDF {
public:
  SpherePDF() = default;

  double value(const Vec3 &direction) const override {
    return 1.0 / (4.0 * PI);
  }

  Vec3 generate() const override { return random_unit_vector(); }

  // JSON serialization method.
  std::string json() const override {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"SpherePDF\",";
    oss << "\"address\":\"" << this << "\"";
    oss << "}";
    return oss.str();
  }
};

// Cosine-weighted hemisphere sampling PDF.
// Optimized for cosine-weighted hemisphere sampling.
class alignas(16) CosinePDF : public PDF {
public:
  CosinePDF(const Vec3 &w) : m_uvw(w) {}

  double value(const Vec3 &direction) const override {
    double cosine_theta = dot_product(unit_vector(direction), m_uvw.w());
    return std::fmax(0, cosine_theta / PI);
  }

  Vec3 generate() const override {
    return m_uvw.transform(random_cosine_direction());
  }

  const ONB &get_onb() const { return m_uvw; }

  // JSON serialization method.
  std::string json() const override {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"CosinePDF\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"uvw\":" << m_uvw.json();
    oss << "}";
    return oss.str();
  }

private:
  ONB m_uvw;
};

// PDF for sampling based on a hittable object.
// Optimized for object-based importance sampling.
class alignas(16) HittablePDF : public PDF {
public:
  HittablePDF(const Hittable &objects, const Point3 &origin)
      : m_objects(objects), m_origin(origin) {}

  double value(const Vec3 &direction) const override {
    return m_objects.pdf_value(m_origin, direction);
  }

  Vec3 generate() const override { return m_objects.random(m_origin); }

  const Hittable &get_objects() const { return m_objects; }

  Point3 get_origin() const { return m_origin; }

  // JSON serialization method.
  std::string json() const override {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"HittablePDF\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"origin\":" << m_origin.json() << ",";
    oss << "\"objects\":" << m_objects.json();
    oss << "}";
    return oss.str();
  }

private:
  // Hot data: origin point used in sampling calculations.

  // Sampling origin point.
  Point3 m_origin;

  // Warm data: object reference for sampling.

  // Objects to sample from.
  const Hittable &m_objects;
};

// Mixture of two PDFs.
// Optimized for PDF mixing operations.
class alignas(16) MixturePDF : public PDF {
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

  PDFPtr get_p0() const { return m_p[0]; }

  PDFPtr get_p1() const { return m_p[1]; }

  // JSON serialization method.
  std::string json() const override {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"MixturePDF\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"p0\":" << (m_p[0] ? m_p[0]->json() : "null") << ",";
    oss << "\"p1\":" << (m_p[1] ? m_p[1]->json() : "null");
    oss << "}";
    return oss.str();
  }

private:
  PDFPtr m_p[2];
};
