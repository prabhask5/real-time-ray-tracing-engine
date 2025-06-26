#pragma once

#ifdef USE_CUDA

#include <stdexcept>

#include "../../core/HittableConversions.cuh"
#include "ONBConversions.cuh"
#include "PDF.cuh"
#include "PDF.hpp"
#include "PDFTypes.hpp"
#include "Vec3Conversions.cuh"

// Convert CPU PDF to CUDA PDF POD struct with type detection.
inline CudaPDF cpu_to_cuda_pdf(const PDF &cpu_pdf) {
  // Use runtime type identification to determine PDF type.
  if (auto sphere_pdf = dynamic_cast<const SpherePDF *>(&cpu_pdf)) {
    return cuda_make_pdf_sphere();
  } else if (auto cosine_pdf = dynamic_cast<const CosinePDF *>(&cpu_pdf)) {
    ONB cpu_onb = cosine_pdf->get_onb();
    return cuda_make_pdf_cosine(cpu_to_cuda_onb(cpu_onb));
  } else if (auto hittable_pdf = dynamic_cast<const HittablePDF *>(&cpu_pdf)) {
    // Convert hittable objects and origin.
    const Hittable &cpu_objects = hittable_pdf->get_objects();
    Point3 cpu_origin = hittable_pdf->get_origin();

    // Create CUDA hittable and store pointer.
    CudaHittable *cuda_hittable = new CudaHittable();
    *cuda_hittable = cpu_to_cuda_hittable(cpu_objects);
    CudaPoint3 cuda_origin = cpu_to_cuda_vec3(cpu_origin);

    return cuda_make_pdf_hittable(cuda_hittable, cuda_origin);
  } else if (auto mixture_pdf = dynamic_cast<const MixturePDF *>(&cpu_pdf)) {
    // Convert both PDFs in the mixture.
    PDFPtr cpu_pdf0 = mixture_pdf->get_p0();
    PDFPtr cpu_pdf1 = mixture_pdf->get_p1();

    CudaPDF *cuda_pdf0 = new CudaPDF();
    CudaPDF *cuda_pdf1 = new CudaPDF();
    *cuda_pdf0 = cpu_to_cuda_pdf(*cpu_pdf0);
    *cuda_pdf1 = cpu_to_cuda_pdf(*cpu_pdf1);

    return cuda_make_pdf_mixture(&cuda_pdf0, &cuda_pdf1);
  } else {
    throw std::runtime_error(
        "PDFConversions.cuh::cpu_to_cuda_pdf: Unknown PDF type encountered "
        "during CPU to CUDA PDF conversion. Unable to convert unrecognized PDF "
        "object.");
  }
}

#endif // USE_CUDA