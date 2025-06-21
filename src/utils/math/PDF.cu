#ifdef USE_CUDA

#include "../../core/Hittable.cuh"
#include "PDF.cuh"

__device__ double CudaHittablePDF::value(const CudaVec3 &direction) const {
  return hittable_pointer->pdf_value(origin, direction);
}

__device__ CudaVec3 CudaHittablePDF::generate(curandState *state) const {
  return hittable_pointer->random(origin, state);
}

__device__ double CudaMixturePDF::value(const CudaVec3 &direction) const {
  return 0.5 * pdf0_pointer->value(direction) +
         0.5 * pdf1_pointer->value(direction);
}

__device__ CudaVec3 CudaMixturePDF::generate(curandState *state) const {
  if (cuda_random_double(state) < 0.5)
    return pdf0_pointer->generate(state);
  else
    return pdf1_pointer->generate(state);
}

__device__ double CudaPDF::value(const CudaVec3 &direction) const {
  switch (type) {
  case CudaPDFType::CUDA_PDF_SPHERE:
    return sphere->value(direction);
  case CudaPDFType::CUDA_PDF_COSINE:
    return cosine->value(direction);
  case CudaPDFType::CUDA_PDF_HITTABLE:
    return hittable->value(direction);
  case CudaPDFType::CUDA_PDF_MIXTURE:
    return mixture->value(direction);
  default:
    return 0.0;
  }
}

__device__ CudaVec3 CudaPDF::generate(curandState *state) const {
  switch (type) {
  case CudaPDFType::CUDA_PDF_SPHERE:
    return sphere->generate(state);
  case CudaPDFType::CUDA_PDF_COSINE:
    return cosine->generate(state);
  case CudaPDFType::CUDA_PDF_HITTABLE:
    return hittable->generate(state);
  case CudaPDFType::CUDA_PDF_MIXTURE:
    return mixture->generate(state);
  default:
    return CudaVec3(1, 0, 0);
  }
}

__device__ CudaPDF cuda_make_sphere_pdf() {
  CudaPDF pdf;
  pdf.type = CudaPDFType::CUDA_PDF_SPHERE;
  pdf.sphere = new CudaSpherePDF();
  return pdf;
}

__device__ CudaPDF cuda_make_cosine_pdf(const CudaVec3 &w) {
  CudaPDF pdf;
  pdf.type = CudaPDFType::CUDA_PDF_COSINE;
  pdf.cosine = new CudaCosinePDF(w);
  return pdf;
}

__device__ CudaPDF cuda_make_hittable_pdf(const CudaHittable *hittable_pointer,
                                          const CudaPoint3 &origin) {
  CudaPDF pdf;
  pdf.type = CudaPDFType::CUDA_PDF_HITTABLE;
  pdf.hittable = new CudaHittablePDF(hittable_pointer, origin);
  return pdf;
}

__device__ CudaPDF cuda_make_mixture_pdf(const CudaPDF *pdf0_pointer,
                                         const CudaPDF *pdf1_pointer) {
  CudaPDF pdf;
  pdf.type = CudaPDFType::CUDA_PDF_MIXTURE;
  pdf.mixture = new CudaMixturePDF(pdf0_pointer, pdf1_pointer);
  return pdf;
}

#endif // USE_CUDA