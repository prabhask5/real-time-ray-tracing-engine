#pragma once

#ifdef USE_CUDA

// Forward declaration to avoid circular dependency
struct CudaHittable;
#include "ONB.cuh"
#include "Utility.cuh"
#include "Vec3.cuh"
#include "Vec3Utility.cuh"
#include <curand_kernel.h>

// PDF types enumeration for manual dispatch.
enum class CudaPDFType {
  CUDA_PDF_SPHERE,
  CUDA_PDF_COSINE,
  CUDA_PDF_HITTABLE,
  CUDA_PDF_MIXTURE
};

// Forward declarations.
struct CudaPDF;

// Uniform sampling over the unit sphere.
struct CudaSpherePDF {
  __host__ __device__ CudaSpherePDF() {} // Default constructor.

  __device__ double value(const CudaVec3 &direction) const {
    return 1.0 / (4.0 * CUDA_PI);
  }

  __device__ CudaVec3 generate(curandState *state) const {
    return cuda_random_unit_vector(state);
  }
};

// Cosine-weighted hemisphere sampling PDF.
struct CudaCosinePDF {
  CudaONB uvw;

  __device__ CudaCosinePDF() {} // Default constructor.
  __host__ __device__ CudaCosinePDF(const CudaVec3 &w) : uvw(w) {}

  __device__ double value(const CudaVec3 &direction) const {
    double cosine_theta =
        cuda_dot_product(cuda_unit_vector(direction), uvw.w());
    return fmax(0.0, cosine_theta / CUDA_PI);
  }

  __device__ CudaVec3 generate(curandState *state) const {
    return uvw.transform(cuda_random_cosine_direction(state));
  }
};

// PDF for sampling based on a hittable object.
struct CudaHittablePDF {
  const CudaHittable *hittable_pointer;
  CudaPoint3 origin;

  __device__ CudaHittablePDF() {} // Default constructor.
  __host__ __device__ CudaHittablePDF(const CudaHittable *_hittable_pointer,
                                      const CudaPoint3 &_origin)
      : hittable_pointer(_hittable_pointer), origin(_origin) {}

  __device__ double value(const CudaVec3 &direction) const;
  __device__ CudaVec3 generate(curandState *state) const;
};

// Mixture of two PDFs.
struct CudaMixturePDF {
  const CudaPDF *pdf0_pointer;
  const CudaPDF *pdf1_pointer;

  __device__ CudaMixturePDF() {} // Default constructor.
  __host__ __device__ CudaMixturePDF(const CudaPDF *_pdf0_pointer,
                                     const CudaPDF *_pdf1_pointer)
      : pdf0_pointer(_pdf0_pointer), pdf1_pointer(_pdf1_pointer) {}

  __device__ double value(const CudaVec3 &direction) const;
  __device__ CudaVec3 generate(curandState *state) const;
};

// Unified PDF data structure using manual dispatch.
struct CudaPDF {
  CudaPDFType type;
  union {
    CudaSpherePDF *sphere;
    CudaCosinePDF *cosine;
    CudaHittablePDF *hittable;
    CudaMixturePDF *mixture;
  };

  __host__ __device__ CudaPDF() {} // Default constructor.

  __device__ double value(const CudaVec3 &direction) const;
  __device__ CudaVec3 generate(curandState *state) const;
};

// Helper constructor functions.
__device__ CudaPDF cuda_make_sphere_pdf();
__device__ CudaPDF cuda_make_cosine_pdf(const CudaVec3 &w);
__device__ CudaPDF cuda_make_hittable_pdf(const CudaHittable *hittable_pointer,
                                          const CudaPoint3 &origin);
__device__ CudaPDF cuda_make_mixture_pdf(const CudaPDF *pdf0_pointer,
                                         const CudaPDF *pdf1_pointer);

#endif // USE_CUDA