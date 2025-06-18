#pragma once

#ifdef USE_CUDA

#include "../../core/Vec3Types.hpp"
#include "ONB.cuh"
#include "Utility.cuh"
#include "Vec3Utility.cuh"
#include <curand_kernel.h>

// PDF types enumeration for manual dispatch.
enum class CudaPDFType { SPHERE = 0, COSINE = 1, HITTABLE = 2, MIXTURE = 3 };

// POD structs for each type.

struct CudaSpherePDF {};

__device__ inline double cuda_sphere_pdf_value() { return 1.0 / (4.0 * PI); }

__device__ inline CudaVec3 cuda_sphere_pdf_generate(curandState *state) {
  return cuda_random_unit_vector(state);
}

struct CudaCosinePDF {
  CudaONB onb;
};

__device__ inline double cuda_cosine_pdf_value(const CudaCosinePDF &pdf,
                                               const CudaVec3 &direction) {
  double cosine =
      cuda_dot_product(cuda_unit_vector(direction), cuda_onb_w(pdf.onb));
  return fmax(0.0, cosine / PI);
}

__device__ inline CudaVec3 cuda_cosine_pdf_generate(const CudaCosinePDF &pdf,
                                                    curandState *state) {
  return cuda_onb_transform(pdf.onb, cuda_random_cosine_direction(state));
}

struct CudaHittablePDF {
  const void *hittable_data;
  CudaPoint3 origin;
};

using CudaHittablePDFValueFn = double (*)(const void *, const CudaPoint3 &,
                                          const CudaVec3 &);
using CudaHittableRandomFn = CudaVec3 (*)(const void *, const CudaPoint3 &,
                                          curandState *);

__device__ inline double cuda_hittable_pdf_value(const CudaHittablePDF &pdf,
                                                 const CudaVec3 &direction) {
  const CudaHittablePDFValueFn *vtable =
      reinterpret_cast<const CudaHittablePDFValueFn *>(pdf.hittable_data);
  return (*vtable)(pdf.hittable_data, pdf.origin, direction);
}

__device__ inline CudaVec3
cuda_hittable_pdf_generate(const CudaHittablePDF &pdf, curandState *state) {
  const CudaHittableRandomFn *vtable =
      reinterpret_cast<const CudaHittableRandomFn *>(pdf.hittable_data);
  return (*vtable)(pdf.hittable_data, pdf.origin, state);
}

struct CudaMixturePDF {
  CudaPDFType type0;
  CudaPDFType type1;
  void *data0;
  void *data1;
};

__device__ __forceinline__ double
cuda_mixture_pdf_value(const CudaMixturePDF &m, const CudaVec3 &direction) {
  double v0 = cuda_dispatch_pdf_value(m.type0, m.data0, direction);
  double v1 = cuda_dispatch_pdf_value(m.type1, m.data1, direction);
  return 0.5 * v0 + 0.5 * v1;
}

__device__ __forceinline__ CudaVec3
cuda_mixture_pdf_generate(const CudaMixturePDF &m, curandState *state) {
  if (cuda_random_double(state) < 0.5)
    return cuda_dispatch_pdf_generate(m.type0, m.data0, state);
  return cuda_dispatch_pdf_generate(m.type1, m.data1, state);
}

// Unified PDF object.
struct CudaPDF {
  CudaPDFType type;
  union {
    CudaSpherePDF sphere;
    CudaCosinePDF cosine;
    CudaHittablePDF hittable;
    CudaMixturePDF mixture;
  } data;
};

__device__ __forceinline__ double
cuda_dispatch_pdf_value(CudaPDFType type, void *data,
                        const CudaVec3 &direction) {
  switch (type) {
  case CudaPDFType::SPHERE:
    return cuda_sphere_pdf_value();
  case CudaPDFType::COSINE:
    return cuda_cosine_pdf_value(*reinterpret_cast<CudaCosinePDF *>(data),
                                 direction);
  case CudaPDFType::HITTABLE:
    return cuda_hittable_pdf_value(*reinterpret_cast<CudaHittablePDF *>(data),
                                   direction);
  case CudaPDFType::MIXTURE:
    return cuda_mixture_pdf_value(*reinterpret_cast<CudaMixturePDF *>(data),
                                  direction);
  }
  return 0.0;
}

__device__ __forceinline__ CudaVec3
cuda_dispatch_pdf_generate(CudaPDFType type, void *data, curandState *state) {
  switch (type) {
  case CudaPDFType::SPHERE:
    return cuda_sphere_pdf_generate(state);
  case CudaPDFType::COSINE:
    return cuda_cosine_pdf_generate(*reinterpret_cast<CudaCosinePDF *>(data),
                                    state);
  case CudaPDFType::HITTABLE:
    return cuda_hittable_pdf_generate(
        *reinterpret_cast<CudaHittablePDF *>(data), state);
  case CudaPDFType::MIXTURE:
    return cuda_mixture_pdf_generate(*reinterpret_cast<CudaMixturePDF *>(data),
                                     state);
  }
  return CudaVec3(0, 0, 1);
}

// Main entry points.

__device__ __forceinline__ double cuda_pdf_value(const CudaPDF &pdf,
                                                 const CudaVec3 &direction) {
  switch (pdf.type) {
  case CudaPDFType::SPHERE:
    return cuda_sphere_pdf_value();
  case CudaPDFType::COSINE:
    return cuda_cosine_pdf_value(pdf.data.cosine, direction);
  case CudaPDFType::HITTABLE:
    return cuda_hittable_pdf_value(pdf.data.hittable, direction);
  case CudaPDFType::MIXTURE:
    return cuda_mixture_pdf_value(pdf.data.mixture, direction);
  }
  return 0.0;
}

__device__ __forceinline__ CudaVec3 cuda_pdf_generate(const CudaPDF &pdf,
                                                      curandState *state) {
  switch (pdf.type) {
  case CudaPDFType::SPHERE:
    return cuda_sphere_pdf_generate(state);
  case CudaPDFType::COSINE:
    return cuda_cosine_pdf_generate(pdf.data.cosine, state);
  case CudaPDFType::HITTABLE:
    return cuda_hittable_pdf_generate(pdf.data.hittable, state);
  case CudaPDFType::MIXTURE:
    return cuda_mixture_pdf_generate(pdf.data.mixture, state);
  }
  return CudaVec3(0, 0, 1);
}

// Constructors.

__device__ inline CudaPDF cuda_make_sphere_pdf() {
  CudaPDF pdf;
  pdf.type = CudaPDFType::SPHERE;
  return pdf;
}

__device__ inline CudaPDF cuda_make_cosine_pdf(const CudaVec3 &normal) {
  CudaPDF pdf;
  pdf.type = CudaPDFType::COSINE;
  pdf.data.cosine.onb = cuda_onb_from_normal(normal);
  return pdf;
}

__device__ inline CudaPDF cuda_make_hittable_pdf(const void *hittable_data,
                                                 const CudaPoint3 &origin) {
  CudaPDF pdf;
  pdf.type = CudaPDFType::HITTABLE;
  pdf.data.hittable.hittable_data = hittable_data;
  pdf.data.hittable.origin = origin;
  return pdf;
}

__device__ inline CudaPDF cuda_make_mixture_pdf(CudaPDFType type0, void *data0,
                                                CudaPDFType type1,
                                                void *data1) {
  CudaPDF pdf;
  pdf.type = CudaPDFType::MIXTURE;
  pdf.data.mixture.type0 = type0;
  pdf.data.mixture.data0 = data0;
  pdf.data.mixture.type1 = type1;
  pdf.data.mixture.data1 = data1;
  return pdf;
}

#endif // USE_CUDA
