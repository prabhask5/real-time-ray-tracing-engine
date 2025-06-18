#pragma once

#ifdef USE_CUDA

#include "../../core/Vec3Types.cuh"
#include "ONB.cuh"
#include "Utility.cuh"
#include "Vec3Utility.cuh"
#include <curand_kernel.h>

// PDF types enumeration for manual dispatch.
enum CudaPDFType {
  CUDA_PDF_SPHERE = 0,
  CUDA_PDF_COSINE = 1,
  CUDA_PDF_HITTABLE = 2,
  CUDA_PDF_MIXTURE = 3
};

// Uniform sampling over the unit sphere.
struct CudaSpherePDF {
  __device__ double value(const CudaVec3 &direction) const {
    return 1.0 / (4.0 * CUDA_PI);
  }

  __device__ CudaVec3 generate(curandState *state) const {
    return cuda_random_unit_vector(state);
  }
};

// Cosine-weighted hemisphere sampling PDF.
struct CudaCosinePDF {
  CudaONB m_uvw;

  __device__ CudaCosinePDF(const CudaVec3 &w) : m_uvw(w) {}

  __device__ double value(const CudaVec3 &direction) const {
    double cosine_theta =
        cuda_dot_product(cuda_unit_vector(direction), m_uvw.w());
    return fmax(0.0, cosine_theta / CUDA_PI);
  }

  __device__ CudaVec3 generate(curandState *state) const {
    return m_uvw.transform(cuda_random_cosine_direction(state));
  }
};

// PDF for sampling based on a hittable object.
struct CudaHittablePDF {
  const void *m_objects_data;
  CudaPoint3 m_origin;

  __device__ CudaHittablePDF(const void *objects_data, const CudaPoint3 &origin)
      : m_objects_data(objects_data), m_origin(origin) {}

  __device__ double value(const CudaVec3 &direction) const {
    // TODO: Implement hittable PDF value calculation
    return 1.0 / (4.0 * CUDA_PI);
  }

  __device__ CudaVec3 generate(curandState *state) const {
    // TODO: Implement hittable PDF generation
    return cuda_random_unit_vector(state);
  }
};

// Mixture of two PDFs.
struct CudaMixturePDF {
  CudaPDFType m_type0, m_type1;
  void *m_data0;
  void *m_data1;

  __device__ CudaMixturePDF(CudaPDFType type0, void *data0, CudaPDFType type1,
                            void *data1)
      : m_type0(type0), m_data0(data0), m_type1(type1), m_data1(data1) {}

  __device__ double value(const CudaVec3 &direction) const {
    double v0 = cuda_dispatch_pdf_value(m_type0, m_data0, direction);
    double v1 = cuda_dispatch_pdf_value(m_type1, m_data1, direction);
    return 0.5 * v0 + 0.5 * v1;
  }

  __device__ CudaVec3 generate(curandState *state) const {
    if (cuda_random_double(state) < 0.5)
      return cuda_dispatch_pdf_generate(m_type0, m_data0, state);
    return cuda_dispatch_pdf_generate(m_type1, m_data1, state);
  }
};

// Unified PDF data structure using manual dispatch
struct CudaPDF {
  CudaPDFType type;
  union {
    CudaSpherePDF sphere;
    CudaCosinePDF cosine;
    CudaHittablePDF hittable;
    CudaMixturePDF mixture;
  } data;

  __device__ inline double value(const CudaVec3 &direction) {
    switch (type) {
    case CudaPDFType::CUDA_PDF_SPHERE:
      return data.sphere.value(direction);
    case CudaPDFType::CUDA_PDF_COSINE:
      return data.cosine.value(direction);
    case CudaPDFType::CUDA_PDF_HITTABLE:
      return data.hittable.value(direction);
    case CudaPDFType::CUDA_PDF_MIXTURE:
      return data.mixture.value(direction);
    }
    return 0.0;
  }

  __device__ inline CudaVec3 generate(curandState *state) {
    switch (type) {
    case CudaPDFType::CUDA_PDF_SPHERE:
      return data.sphere.generate(state);
    case CudaPDFType::CUDA_PDF_COSINE:
      return data.cosine.generate(state);
    case CudaPDFType::CUDA_PDF_HITTABLE:
      return data.hittable.generate(state);
    case CudaPDFType::CUDA_PDF_MIXTURE:
      return data.mixture.generate(state);
    }
    return CudaVec3(0, 0, 1);
  }
};

// Manual dispatch functions for PDF operations
__device__ inline double cuda_dispatch_pdf_value(CudaPDFType type, void *data,
                                                 const CudaVec3 &direction) {
  switch (type) {
  case CudaPDFType::CUDA_PDF_SPHERE:
    return reinterpret_cast<CudaSpherePDF *>(data)->value(direction);
  case CudaPDFType::CUDA_PDF_COSINE:
    return reinterpret_cast<CudaCosinePDF *>(data)->value(direction);
  case CudaPDFType::CUDA_PDF_HITTABLE:
    return reinterpret_cast<CudaHittablePDF *>(data)->value(direction);
  case CudaPDFType::CUDA_PDF_MIXTURE:
    return reinterpret_cast<CudaMixturePDF *>(data)->value(direction);
  }
  return 0.0;
}

__device__ inline CudaVec3
cuda_dispatch_pdf_generate(CudaPDFType type, void *data, curandState *state) {
  switch (type) {
  case CudaPDFType::CUDA_PDF_SPHERE:
    return reinterpret_cast<CudaSpherePDF *>(data)->generate(state);
  case CudaPDFType::CUDA_PDF_COSINE:
    return reinterpret_cast<CudaCosinePDF *>(data)->generate(state);
  case CudaPDFType::CUDA_PDF_HITTABLE:
    return reinterpret_cast<CudaHittablePDF *>(data)->generate(state);
  case CudaPDFType::CUDA_PDF_MIXTURE:
    return reinterpret_cast<CudaMixturePDF *>(data)->generate(state);
  }
  return CudaVec3(0, 0, 1);
}

// Helper constructor functions
__device__ inline CudaPDF cuda_make_sphere_pdf() {
  CudaPDF pdf;
  pdf.type = CudaPDFType::CUDA_PDF_SPHERE;
  pdf.data.sphere = CudaSpherePDF();
  return pdf;
}

__device__ inline CudaPDF cuda_make_cosine_pdf(const CudaVec3 &normal) {
  CudaPDF pdf;
  pdf.type = CudaPDFType::CUDA_PDF_COSINE;
  pdf.data.cosine = CudaCosinePDF(normal);
  return pdf;
}

__device__ inline CudaPDF cuda_make_hittable_pdf(const void *objects_data,
                                                 const CudaPoint3 &origin) {
  CudaPDF pdf;
  pdf.type = CudaPDFType::CUDA_PDF_HITTABLE;
  pdf.data.hittable = CudaHittablePDF(objects_data, origin);
  return pdf;
}

__device__ inline CudaPDF cuda_make_mixture_pdf(CudaPDFType type0, void *data0,
                                                CudaPDFType type1,
                                                void *data1) {
  CudaPDF pdf;
  pdf.type = CudaPDFType::CUDA_PDF_MIXTURE;
  pdf.data.mixture = CudaMixturePDF(type0, data0, type1, data1);
  return pdf;
}

#endif // USE_CUDA