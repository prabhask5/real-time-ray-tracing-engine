#pragma once

#include "Ray.hpp"
#include <PDFTypes.hpp>
#include <Vec3.hpp>
#include <Vec3Types.hpp>

// Used in a ray tracer to describe the result of a material's scatter()
// function — that is, how a ray interacts with a surface (reflection,
// transmission, etc).
class ScatterRecord {
public:
  // Represents how much the surface attenuates (reduces or filters) light
  // passing through it. Think of it as the color tint the surface gives to the
  // scattered ray. If it's (1, 1, 1) → no change. If it's (0.5, 0, 0) → dims
  // everything and leaves only red.
  Color attenuation;

  // A smart pointer (shared_ptr) to a probability density function object.
  // This tells the renderer how to randomly sample the direction of scattered
  // rays. Used when importance sampling (e.g. cosine-weighted scattering for
  // Lambertian surfaces or sampling toward lights).
  PDFPtr pdf_ptr;

  // If this is true, the material wants to skip sampling a direction from the
  // PDF and instead explicitly provides the scattered ray in skip_pdf_ray. This
  // is useful for perfect specular reflection (like a mirror), where the
  // direction is deterministic.
  bool skip_pdf;

  // Only used if skip_pdf is true.
  // Represents the actual scattered ray (for deterministic materials like
  // mirrors or dielectrics).
  Ray skip_pdf_ray;
};