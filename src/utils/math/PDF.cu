#ifdef USE_CUDA

#include "../../core/Hittable.cuh"
#include "../memory/CudaMemoryUtility.cuh"
#include "PDF.cuh"
#include <iomanip>
#include <sstream>

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

// JSON serialization functions for CUDA PDF structures.
std::string cuda_json_sphere_pdf(const CudaSpherePDF &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaSpherePDF\",";
  oss << "\"address\":\"" << &obj << "\"";
  oss << "}";
  return oss.str();
}

std::string cuda_json_cosine_pdf(const CudaCosinePDF &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaCosinePDF\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"uvw\":" << cuda_json_onb(obj.uvw);
  oss << "}";
  return oss.str();
}

std::string cuda_json_hittable_pdf(const CudaHittablePDF &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaHittablePDF\",";
  oss << "\"address\":\"" << &obj << "\",";
  if (obj.hittable) {
    CudaHittable host_hittable;
    cudaMemcpyDeviceToHostSafe(&host_hittable, obj.hittable, 1);
    oss << "\"hittable\":" << cuda_json_hittable(host_hittable) << ",";
  } else {
    oss << "\"hittable\":null,";
  }
  oss << "\"origin\":" << cuda_json_vec3(obj.origin);
  oss << "}";
  return oss.str();
}

std::string cuda_json_mixture_pdf(const CudaMixturePDF &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaMixturePDF\",";
  oss << "\"address\":\"" << &obj << "\",";
  if (obj.pdf0) {
    CudaPDF host_pdf0;
    cudaMemcpyDeviceToHostSafe(&host_pdf0, obj.pdf0, 1);
    oss << "\"pdf0\":" << cuda_json_pdf(host_pdf0) << ",";
  } else {
    oss << "\"pdf0\":null,";
  }
  if (obj.pdf1) {
    CudaPDF host_pdf1;
    cudaMemcpyDeviceToHostSafe(&host_pdf1, obj.pdf1, 1);
    oss << "\"pdf1\":" << cuda_json_pdf(host_pdf1);
  } else {
    oss << "\"pdf1\":null";
  }
  oss << "}";
  return oss.str();
}

std::string cuda_json_pdf(const CudaPDF &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaPDF\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"pdf_type\":";
  switch (obj.type) {
  case CudaPDFType::CUDA_PDF_SPHERE:
    oss << "\"SPHERE\",";
    oss << "\"sphere\":" << cuda_json_sphere_pdf(obj.sphere);
    break;
  case CudaPDFType::CUDA_PDF_COSINE:
    oss << "\"COSINE\",";
    oss << "\"cosine\":" << cuda_json_cosine_pdf(obj.cosine);
    break;
  case CudaPDFType::CUDA_PDF_HITTABLE:
    oss << "\"HITTABLE\",";
    oss << "\"hittable\":" << cuda_json_hittable_pdf(obj.hittable);
    break;
  case CudaPDFType::CUDA_PDF_MIXTURE:
    oss << "\"MIXTURE\",";
    oss << "\"mixture\":" << cuda_json_mixture_pdf(obj.mixture);
    break;
  }
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA