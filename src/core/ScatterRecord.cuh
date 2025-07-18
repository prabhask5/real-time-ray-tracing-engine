#pragma once

#ifdef USE_CUDA

#include "../utils/math/PDF.cuh"
#include "Ray.cuh"
#include "Vec3Types.cuh"
#include <climits>
#include <iomanip>
#include <sstream>

// Used in a ray tracer to describe the result of a material's scatter()
// function — that is, how a ray interacts with a surface (reflection,
// transmission, etc).
struct CudaScatterRecord {
  // Represents how much the surface attenuates (reduces or filters) light
  // passing through it. Think of it as the color tint the surface gives to the
  // scattered ray. If it's (1, 1, 1) → no change. If it's (0.5, 0, 0) → dims
  // everything and leaves only red.
  CudaColor attenuation;

  // Probability density function for importance sampling.
  // This tells the renderer how to randomly sample the direction of scattered
  // rays. Used when importance sampling (e.g. cosine-weighted scattering for
  // Lambertian surfaces or sampling toward lights).
  CudaPDF pdf;

  // If this is true, the material wants to skip sampling a direction from the
  // PDF and instead explicitly provides the scattered ray in skip_pdf_ray. This
  // is useful for perfect specular reflection (like a mirror), where the
  // direction is deterministic.
  bool skip_pdf;

  // Only used if skip_pdf is true.
  // Represents the actual scattered ray (for deterministic materials like
  // mirrors or dielectrics).
  CudaRay skip_pdf_ray;
};

// JSON serialization function for CudaScatterRecord.
inline std::string cuda_json_scatter_record(const CudaScatterRecord &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaScatterRecord\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"attenuation\":" << cuda_json_vec3(obj.attenuation) << ",";
  oss << "\"pdf\":\"" << &obj.pdf << "\",";
  oss << "\"skip_pdf\":" << (obj.skip_pdf ? "true" : "false") << ",";
  oss << "\"skip_pdf_ray\":" << cuda_json_ray(obj.skip_pdf_ray);
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA