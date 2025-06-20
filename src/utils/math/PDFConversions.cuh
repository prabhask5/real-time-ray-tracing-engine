#pragma once

#ifdef USE_CUDA

#include "../../core/HittableConversions.cuh"
#include "ONBConversions.cuh"
#include "PDF.cuh"
#include "PDFTypes.hpp"
#include "Vec3Conversions.cuh"

// Forward declarations for CPU PDF types.
class PDF;
class SpherePDF;
class CosinePDF;
class HittablePDF;
class MixturePDF;

// Convert CPU PDF to CUDA PDF with type detection.
inline CudaPDF cpu_to_cuda_pdf(const PDF &cpu_pdf) {
  CudaPDF cuda_pdf;

  // Use runtime type identification to determine PDF type.
  if (auto sphere_pdf = dynamic_cast<const SpherePDF *>(&cpu_pdf)) {
    cuda_pdf.type = CudaPDFType::CUDA_PDF_SPHERE;
    cuda_pdf.data.sphere = CudaSpherePDF();
  } else if (auto cosine_pdf = dynamic_cast<const CosinePDF *>(&cpu_pdf)) {
    cuda_pdf.type = CudaPDFType::CUDA_PDF_COSINE;
    Vec3 cpu_normal = cosine_pdf->get_onb().w();
    cuda_pdf.data.cosine = CudaCosinePDF(cpu_to_cuda_vec3(cpu_normal));
  } else if (auto hittable_pdf = dynamic_cast<const HittablePDF *>(&cpu_pdf)) {
    cuda_pdf.type = CudaPDFType::CUDA_PDF_HITTABLE;
    // Convert hittable objects and origin.
    auto cpu_objects = hittable_pdf->get_objects();
    Point3 cpu_origin = hittable_pdf->get_origin();

    // Create CUDA hittable and store pointer.
    CudaHittable *cuda_objects = new CudaHittable();
    *cuda_objects = cpu_to_cuda_hittable(cpu_objects);
    CudaPoint3 cuda_origin = cpu_to_cuda_vec3(cpu_origin);

    cuda_pdf.data.hittable = CudaHittablePDF(cuda_objects, cuda_origin);
  } else if (auto mixture_pdf = dynamic_cast<const MixturePDF *>(&cpu_pdf)) {
    cuda_pdf.type = CudaPDFType::CUDA_PDF_MIXTURE;

    // Convert both PDFs in the mixture.
    auto cpu_pdf0 = mixture_pdf->get_p0();
    auto cpu_pdf1 = mixture_pdf->get_p1();

    CudaPDF *cuda_pdf0 = new CudaPDF();
    CudaPDF *cuda_pdf1 = new CudaPDF();
    *cuda_pdf0 = cpu_to_cuda_pdf(cpu_pdf0);
    *cuda_pdf1 = cpu_to_cuda_pdf(cpu_pdf1);

    cuda_pdf.data.mixture = CudaMixturePDF(cuda_pdf0->type, &cuda_pdf0->data,
                                           cuda_pdf1->type, &cuda_pdf1->data);
  } else {
    // Default to sphere PDF.
    cuda_pdf.type = CudaPDFType::CUDA_PDF_SPHERE;
    cuda_pdf.data.sphere = CudaSpherePDF();
  }

  return cuda_pdf;
}

// Convert CUDA PDF to CPU PDF.
inline PDFPtr cuda_to_cpu_pdf(const CudaPDF &cuda_pdf) {
  switch (cuda_pdf.type) {
  case CudaPDFType::CUDA_PDF_SPHERE:
    return std::make_shared<SpherePDF>();

  case CudaPDFType::CUDA_PDF_COSINE: {
    Vec3 cpu_normal = cuda_to_cpu_vec3(cuda_pdf.data.cosine.uvw.w());
    return std::make_shared<CosinePDF>(cpu_normal);
  }

  case CudaPDFType::CUDA_PDF_HITTABLE: {
    // Convert back to CPU hittable and origin.
    const CudaHittable *cuda_objects = reinterpret_cast<const CudaHittable *>(
        cuda_pdf.data.hittable.objects_data);
    auto cpu_objects = cuda_to_cpu_hittable(*cuda_objects);
    Point3 cpu_origin = cuda_to_cpu_vec3(cuda_pdf.data.hittable.origin);

    return std::make_shared<HittablePDF>(cpu_objects, cpu_origin);
  }

  case CudaPDFType::CUDA_PDF_MIXTURE: {
    // Convert both PDFs back to CPU.
    const CudaPDF *cuda_pdf0 =
        reinterpret_cast<const CudaPDF *>(cuda_pdf.data.mixture.data0);
    const CudaPDF *cuda_pdf1 =
        reinterpret_cast<const CudaPDF *>(cuda_pdf.data.mixture.data1);

    auto cpu_pdf0 = cuda_to_cpu_pdf(*cuda_pdf0);
    auto cpu_pdf1 = cuda_to_cpu_pdf(*cuda_pdf1);

    return std::make_shared<MixturePDF>(cpu_pdf0, cpu_pdf1);
  }

  default:
    return std::make_shared<SpherePDF>();
  }
}

// Memory management for PDF objects.
inline void cleanup_cuda_pdf(CudaPDF &cuda_pdf) {
  switch (cuda_pdf.type) {
  case CudaPDFType::CUDA_PDF_HITTABLE:
    if (cuda_pdf.data.hittable.objects_data) {
      delete reinterpret_cast<const CudaHittable *>(
          cuda_pdf.data.hittable.objects_data);
      cuda_pdf.data.hittable.objects_data = nullptr;
    }
    break;

  case CudaPDFType::CUDA_PDF_MIXTURE:
    if (cuda_pdf.data.mixture.data0) {
      delete reinterpret_cast<CudaPDF *>(cuda_pdf.data.mixture.data0);
      cuda_pdf.data.mixture.data0 = nullptr;
    }
    if (cuda_pdf.data.mixture.data1) {
      delete reinterpret_cast<CudaPDF *>(cuda_pdf.data.mixture.data1);
      cuda_pdf.data.mixture.data1 = nullptr;
    }
    break;

  default:
    // Other PDF types don't need special cleanup.
    break;
  }
}

#endif // USE_CUDA