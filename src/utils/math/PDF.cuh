#pragma once

#ifdef USE_CUDA

#include "ONB.cuh"
#include "Utility.cuh"
#include "Vec3.cuh"
#include "Vec3Utility.cuh"
#include <curand_kernel.h>

// Forward declaration to avoid circular dependency.
struct CudaHittable;

// PDF types enumeration for manual dispatch.
enum class CudaPDFType {
  CUDA_PDF_SPHERE,
  CUDA_PDF_COSINE,
  CUDA_PDF_HITTABLE,
  CUDA_PDF_MIXTURE
};

// Forward declarations.
struct CudaPDF;

// POD struct for uniform sampling over the unit sphere.
struct CudaSpherePDF {};

// SpherePDF utility functions.
__device__ inline double cuda_sphere_pdf_value(const CudaSpherePDF &pdf,
                                               const CudaVec3 &direction) {
  return 1.0 / (4.0 * CUDA_PI);
}

__device__ inline CudaVec3 cuda_sphere_pdf_generate(const CudaSpherePDF &pdf,
                                                    curandState *state) {
  return cuda_vec3_random_unit_vector(state);
}

// POD struct for cosine-weighted hemisphere sampling PDF.
struct CudaCosinePDF {
  CudaONB uvw;
};

// CosinePDF initialization functions.
__device__ inline CudaCosinePDF cuda_make_cosine_pdf(const CudaVec3 &w) {
  CudaCosinePDF pdf;
  pdf.uvw = cuda_make_onb(w);
  return pdf;
}

__host__ __device__ inline CudaCosinePDF
cuda_make_cosine_pdf(const CudaONB &uvw) {
  CudaCosinePDF pdf;
  pdf.uvw = uvw;
  return pdf;
}

// CosinePDF utility functions.
__device__ inline double cuda_cosine_pdf_value(const CudaCosinePDF &pdf,
                                               const CudaVec3 &direction) {
  double cosine_theta = cuda_vec3_dot_product(cuda_vec3_unit_vector(direction),
                                              cuda_onb_w(pdf.uvw));
  return fmax(0.0, cosine_theta / CUDA_PI);
}

__device__ inline CudaVec3 cuda_cosine_pdf_generate(const CudaCosinePDF &pdf,
                                                    curandState *state) {
  return cuda_onb_transform(pdf.uvw, cuda_vec3_random_cosine_direction(state));
}

// POD struct for sampling based on a hittable object.
struct CudaHittablePDF {
  const CudaHittable *hittable;
  CudaPoint3 origin;
};

// HittablePDF initialization function.
__host__ __device__ inline CudaHittablePDF
cuda_make_hittable_pdf(const CudaHittable *hittable, const CudaPoint3 &origin) {
  CudaHittablePDF pdf;
  pdf.hittable = hittable;
  pdf.origin = origin;
  return pdf;
}

// Forward declarations for HittablePDF utility functions.
__device__ double cuda_hittable_pdf_value(const CudaHittablePDF &pdf,
                                          const CudaVec3 &direction);
__device__ CudaVec3 cuda_hittable_pdf_generate(const CudaHittablePDF &pdf,
                                               curandState *state);

// POD struct for mixture of two PDFs.
struct CudaMixturePDF {
  const CudaPDF *pdf0;
  const CudaPDF *pdf1;
};

// MixturePDF initialization function.
__host__ __device__ inline CudaMixturePDF
cuda_make_mixture_pdf(const CudaPDF *pdf0, const CudaPDF *pdf1) {
  CudaMixturePDF pdf;
  pdf.pdf0 = pdf0;
  pdf.pdf1 = pdf1;
  return pdf;
}

// Forward declarations for MixturePDF utility functions.
__device__ double cuda_mixture_pdf_value(const CudaMixturePDF &pdf,
                                         const CudaVec3 &direction);
__device__ CudaVec3 cuda_mixture_pdf_generate(const CudaMixturePDF &pdf,
                                              curandState *state);

// POD struct for unified PDF data structure using manual dispatch.
struct CudaPDF {
  CudaPDFType type;
  union {
    CudaSpherePDF sphere;
    CudaCosinePDF cosine;
    CudaHittablePDF hittable;
    CudaMixturePDF mixture;
  };
};

// PDF utility functions.
__device__ double cuda_pdf_value(const CudaPDF &pdf, const CudaVec3 &direction);
__device__ CudaVec3 cuda_pdf_generate(const CudaPDF &pdf, curandState *state);

// PDF initialization functions.
__device__ inline CudaPDF cuda_make_pdf_sphere() {
  CudaPDF pdf;
  pdf.type = CudaPDFType::CUDA_PDF_SPHERE;
  pdf.sphere = CudaSpherePDF{};
  return pdf;
}

__device__ inline CudaPDF cuda_make_pdf_cosine(const CudaVec3 &w) {
  CudaPDF pdf;
  pdf.type = CudaPDFType::CUDA_PDF_COSINE;
  pdf.cosine = cuda_make_cosine_pdf(w);
  return pdf;
}

__device__ inline CudaPDF cuda_make_pdf_hittable(const CudaHittable *hittable,
                                                 const CudaPoint3 &origin) {
  CudaPDF pdf;
  pdf.type = CudaPDFType::CUDA_PDF_HITTABLE;
  pdf.hittable = cuda_make_hittable_pdf(hittable, origin);
  return pdf;
}

__device__ inline CudaPDF cuda_make_pdf_mixture(const CudaPDF *pdf0,
                                                const CudaPDF *pdf1) {
  CudaPDF pdf;
  pdf.type = CudaPDFType::CUDA_PDF_MIXTURE;
  pdf.mixture = cuda_make_mixture_pdf(pdf0, pdf1);
  return pdf;
}

#endif // USE_CUDA