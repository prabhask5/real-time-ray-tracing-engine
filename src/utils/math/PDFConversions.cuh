#pragma once

#ifdef USE_CUDA

#include "../../core/HittableConversions.cuh"
#include "ONBConversions.cuh"
#include "PDF.cuh"
#include "PDF.hpp"
#include "PDFTypes.hpp"
#include "Vec3Conversions.cuh"

// Convert CPU PDF to CUDA PDF with type detection.
inline CudaPDF cpu_to_cuda_pdf(const PDF &cpu_pdf) {
  CudaPDF cuda_pdf;

  // Use runtime type identification to determine PDF type.
  if (PDFPtr sphere_pdf = dynamic_cast<const SpherePDF *>(&cpu_pdf)) {
    cuda_pdf.type = CudaPDFType::CUDA_PDF_SPHERE;
    cuda_pdf.sphere = CudaSpherePDF();
  } else if (PDFPtr cosine_pdf = dynamic_cast<const CosinePDF *>(&cpu_pdf)) {
    cuda_pdf.type = CudaPDFType::CUDA_PDF_COSINE;
    Vec3 cpu_normal = cosine_pdf->get_onb().w();
    cuda_pdf.cosine = CudaCosinePDF(cpu_to_cuda_vec3(cpu_normal));
  } else if (PDFPtr hittable_pdf =
                 dynamic_cast<const HittablePDF *>(&cpu_pdf)) {
    cuda_pdf.type = CudaPDFType::CUDA_PDF_HITTABLE;

    // Convert hittable objects and origin.
    const Hittable &cpu_objects = hittable_pdf->get_objects();
    Point3 cpu_origin = hittable_pdf->get_origin();

    // Create CUDA hittable and store pointer.
    CudaHittable *cuda_hittable_pointer = new CudaHittable();
    *cuda_hittable_pointer = cpu_to_cuda_hittable(*cpu_objects);
    CudaPoint3 cuda_origin = cpu_to_cuda_vec3(cpu_origin);

    cuda_pdf.hittable = CudaHittablePDF(cuda_hittable_pointer, cuda_origin);
  } else if (PDFPtr mixture_pdf = dynamic_cast<const MixturePDF *>(&cpu_pdf)) {
    cuda_pdf.type = CudaPDFType::CUDA_PDF_MIXTURE;

    // Convert both PDFs in the mixture.
    PDFPtr cpu_pdf0 = mixture_pdf->get_p0();
    PDFPtr cpu_pdf1 = mixture_pdf->get_p1();

    CudaPDF *cuda_pdf0 = new CudaPDF();
    CudaPDF *cuda_pdf1 = new CudaPDF();
    *cuda_pdf0 = cpu_to_cuda_pdf(cpu_pdf0);
    *cuda_pdf1 = cpu_to_cuda_pdf(cpu_pdf1);

    cuda_pdf.mixture = CudaMixturePDF(&cuda_pdf0, &cuda_pdf1);
  } else {
    // Fallback to sphere PDF.
    cuda_pdf.type = CudaPDFType::CUDA_PDF_SPHERE;
    cuda_pdf.sphere = CudaSpherePDF();
  }

  return cuda_pdf;
}

// Convert CUDA PDF to CPU PDF.
inline PDFPtr cuda_to_cpu_pdf(const CudaPDF &cuda_pdf) {
  switch (cuda_pdf.type) {
  case CudaPDFType::CUDA_PDF_SPHERE:
    return std::make_shared<SpherePDF>();

  case CudaPDFType::CUDA_PDF_COSINE: {
    Vec3 cpu_normal = cuda_to_cpu_vec3(cuda_pdf.cosine.uvw.w());
    return std::make_shared<CosinePDF>(cpu_normal);
  }

  case CudaPDFType::CUDA_PDF_HITTABLE: {
    // Convert back to CPU hittable and origin.
    const CudaHittable *cuda_hittable_pointer =
        cuda_pdf.hittable.hittable_pointer;

    HittablePtr cpu_objects = cuda_to_cpu_hittable(*cuda_hittable_pointer);
    Point3 cpu_origin = cuda_to_cpu_vec3(cuda_pdf.hittable.origin);

    return std::make_shared<HittablePDF>(cpu_objects, cpu_origin);
  }

  case CudaPDFType::CUDA_PDF_MIXTURE: {
    // Convert both PDFs back to CPU.
    const CudaPDF *cuda_pdf0 = cuda_pdf.mixture.pdf0_pointer;
    const CudaPDF *cuda_pdf1 = cuda_pdf.mixture.pdf1_pointer;

    PDFPtr cpu_pdf0 = cuda_to_cpu_pdf(*cuda_pdf0);
    PDFPtr cpu_pdf1 = cuda_to_cpu_pdf(*cuda_pdf1);

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
    if (cuda_pdf.hittable.hittable_pointer) {
      delete cuda_pdf.hittable.hittable_pointer;
      cuda_pdf.hittable.hittable_pointer = nullptr;
    }
    break;

  case CudaPDFType::CUDA_PDF_MIXTURE:
    if (cuda_pdf.mixture.pdf0_pointer) {
      delete cuda_pdf.mixture.pdf0_pointer;
      cuda_pdf.mixture.pdf0_pointer = nullptr;
    }
    if (cuda_pdf.mixture.pdf1_pointer) {
      delete cuda_pdf.mixture.pdf1_pointer;
      cuda_pdf.mixture.pdf1_pointer = nullptr;
    }
    break;

  default:
    // Other PDF types don't need special cleanup.
    break;
  }
}

#endif // USE_CUDA