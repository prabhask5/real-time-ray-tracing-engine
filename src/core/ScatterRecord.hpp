#pragma once

#include "../utils/math/PDF.hpp"
#include "../utils/math/PDFTypes.hpp"
#include "../utils/math/Vec3.hpp"
#include "Ray.hpp"
#include "Vec3Types.hpp"
#include <iomanip>
#include <sstream>

// Used in a ray tracer to describe the result of a material's scatter()
// function — that is, how a ray interacts with a surface (reflection,
// transmission, etc).
// Memory layout optimized for cache efficiency.
class alignas(16) ScatterRecord {
public:
  // Hot data: most frequently accessed first.

  // Represents how much the surface attenuates (reduces or filters) light
  // passing through it. Think of it as the color tint the surface gives to the
  // scattered ray. If it's (1, 1, 1) → no change. If it's (0.5, 0, 0) → dims
  // everything and leaves only red.
  Color attenuation;

  // Group smaller frequently accessed data together.

  // A smart pointer (shared_ptr) to a probability density function object.
  // This tells the renderer how to randomly sample the direction of scattered
  // rays. Used when importance sampling (e.g. cosine-weighted scattering for
  // Lambertian surfaces or sampling toward lights).
  PDFPtr pdf;

  // If this is true, the material wants to skip sampling a direction from the
  // PDF and instead explicitly provides the scattered ray in skip_pdf_ray. This
  // is useful for perfect specular reflection (like a mirror), where the
  // direction is deterministic.
  bool skip_pdf;

  // Cold data: conditionally used, place last.

  // Only used if skip_pdf is true.
  // Represents the actual scattered ray (for deterministic materials like
  // mirrors or dielectrics).
  Ray skip_pdf_ray;

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"ScatterRecord\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"attenuation\":" << attenuation.json() << ",";
    oss << "\"pdf\":\"" << (pdf ? pdf->json() : "null") << "\",";
    oss << "\"skip_pdf\":" << (skip_pdf ? "true" : "false") << ",";
    oss << "\"skip_pdf_ray\":" << skip_pdf_ray.json();
    oss << "}";
    return oss.str();
  }
};