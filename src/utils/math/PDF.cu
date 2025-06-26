#ifdef USE_CUDA

#include "../../core/Hittable.cuh"
#include "PDF.cuh"

__device__ double cuda_hittable_pdf_value(const CudaHittablePDF &pdf,
                                          const CudaVec3 &direction) {
  return cuda_hittable_pdf_value(*pdf.hittable, pdf.origin, direction);
}

__device__ CudaVec3 cuda_hittable_pdf_generate(const CudaHittablePDF &pdf,
                                               curandState *state) {
  return cuda_hittable_random(*pdf.hittable, pdf.origin, state);
}

__device__ double cuda_mixture_pdf_value(const CudaMixturePDF &pdf,
                                         const CudaVec3 &direction) {
  return 0.5 * cuda_pdf_value(*pdf.pdf0, direction) +
         0.5 * cuda_pdf_value(*pdf.pdf1, direction);
}

__device__ CudaVec3 cuda_mixture_pdf_generate(const CudaMixturePDF &pdf,
                                              curandState *state) {
  if (cuda_random_double(state) < 0.5)
    return cuda_pdf_generate(*pdf.pdf0, state);
  else
    return cuda_pdf_generate(*pdf.pdf1, state);
}

__device__ double cuda_pdf_value(const CudaPDF &pdf,
                                 const CudaVec3 &direction) {
  switch (pdf.type) {
  case CudaPDFType::CUDA_PDF_SPHERE:
    return cuda_sphere_pdf_value(pdf.sphere, direction);
  case CudaPDFType::CUDA_PDF_COSINE:
    return cuda_cosine_pdf_value(pdf.cosine, direction);
  case CudaPDFType::CUDA_PDF_HITTABLE:
    return cuda_hittable_pdf_value(pdf.hittable, direction);
  case CudaPDFType::CUDA_PDF_MIXTURE:
    return cuda_mixture_pdf_value(pdf.mixture, direction);
  default:
    // ERROR: PDF.cu::value - Unknown PDF type in switch statement. This should
    // never happen in well-formed code.
    return 0.0; // Safe fallback for GPU device code.
  }
}

__device__ CudaVec3 cuda_pdf_generate(const CudaPDF &pdf, curandState *state) {
  switch (pdf.type) {
  case CudaPDFType::CUDA_PDF_SPHERE:
    return cuda_sphere_pdf_generate(pdf.sphere, state);
  case CudaPDFType::CUDA_PDF_COSINE:
    return cuda_cosine_pdf_generate(pdf.cosine, state);
  case CudaPDFType::CUDA_PDF_HITTABLE:
    return cuda_hittable_pdf_generate(pdf.hittable, state);
  case CudaPDFType::CUDA_PDF_MIXTURE:
    return cuda_mixture_pdf_generate(pdf.mixture, state);
  default:
    // ERROR: PDF.cu::generate - Unknown PDF type in switch statement. This
    // should never happen in well-formed code.
    return cuda_make_vec3(1, 0, 0); // Safe fallback for GPU device code.
  }
}

#endif // USE_CUDA