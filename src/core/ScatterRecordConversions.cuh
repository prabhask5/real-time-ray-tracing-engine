#pragma once

#ifdef USE_CUDA

#include "../utils/math/PDFConversions.cuh"
#include "../utils/math/Vec3Conversions.cuh"
#include "RayConversions.cuh"
#include "ScatterRecord.cuh"
#include "ScatterRecord.hpp"

// Convert CPU ScatterRecord to CUDA ScatterRecord.
inline CudaScatterRecord
cpu_to_cuda_scatter_record(const ScatterRecord &cpu_scatter_record) {
  CudaScatterRecord cuda_scatter_record;

  // Convert attenuation color.
  cuda_scatter_record.attenuation =
      cpu_to_cuda_vec3(cpu_scatter_record.attenuation);

  // Handle PDF conversion - CPU uses shared_ptr, CUDA uses type enum + raw
  // pointer.
  if (cpu_scatter_record.pdf_ptr != nullptr) {
    auto cuda_pdf = cpu_to_cuda_pdf(*cpu_scatter_record.pdf_ptr);
    cuda_scatter_record.pdf_type = cuda_pdf.type;

    // Allocate and copy PDF data on device.
    switch (cuda_pdf.type) {
    case CudaPDFType::CUDA_PDF_COSINE: {
      CudaCosinePDF *device_pdf = new CudaCosinePDF();
      *device_pdf = cuda_pdf.data.cosine_pdf;
      cuda_scatter_record.pdf_data = (void *)device_pdf;
      break;
    }
    case CudaPDFType::CUDA_PDF_SPHERE: {
      CudaSpherePDF *device_pdf = new CudaSpherePDF();
      *device_pdf = cuda_pdf.data.sphere_pdf;
      cuda_scatter_record.pdf_data = (void *)device_pdf;
      break;
    }
    case CudaPDFType::CUDA_PDF_HITTABLE: {
      CudaHittablePDF *device_pdf = new CudaHittablePDF();
      *device_pdf = cuda_pdf.data.hittable_pdf;
      cuda_scatter_record.pdf_data = (void *)device_pdf;
      break;
    }
    case CudaPDFType::CUDA_PDF_MIXTURE: {
      CudaMixturePDF *device_pdf = new CudaMixturePDF();
      *device_pdf = cuda_pdf.data.mixture_pdf;
      cuda_scatter_record.pdf_data = (void *)device_pdf;
      break;
    }
    default:
      cuda_scatter_record.pdf_data = nullptr;
      break;
    }
  } else {
    cuda_scatter_record.pdf_type = CudaPDFType::CUDA_PDF_COSINE;
    cuda_scatter_record.pdf_data = nullptr;
  }

  // Copy skip_pdf flag.
  cuda_scatter_record.skip_pdf = cpu_scatter_record.skip_pdf;

  // Convert skip_pdf_ray.
  cuda_scatter_record.skip_pdf_ray =
      cpu_to_cuda_ray(cpu_scatter_record.skip_pdf_ray);

  return cuda_scatter_record;
}

// Convert CUDA ScatterRecord to CPU ScatterRecord.
__host__ __device__ inline ScatterRecord
cuda_to_cpu_scatter_record(const CudaScatterRecord &cuda_scatter_record) {
  ScatterRecord cpu_scatter_record;

  // Convert attenuation color.
  cpu_scatter_record.attenuation =
      cuda_to_cpu_vec3(cuda_scatter_record.attenuation);

  // Handle PDF conversion - CUDA uses type enum + raw pointer, CPU uses
  // shared_ptr.
  if (cuda_scatter_record.pdf_data != nullptr) {
    CudaPDF cuda_pdf;
    cuda_pdf.type = cuda_scatter_record.pdf_type;

    switch (cuda_scatter_record.pdf_type) {
    case CudaPDFType::CUDA_PDF_COSINE:
      cuda_pdf.data.cosine_pdf = *reinterpret_cast<const CudaCosinePDF *>(
          cuda_scatter_record.pdf_data);
      break;
    case CudaPDFType::CUDA_PDF_SPHERE:
      cuda_pdf.data.sphere_pdf = *reinterpret_cast<const CudaSpherePDF *>(
          cuda_scatter_record.pdf_data);
      break;
    case CudaPDFType::CUDA_PDF_HITTABLE:
      cuda_pdf.data.hittable_pdf = *reinterpret_cast<const CudaHittablePDF *>(
          cuda_scatter_record.pdf_data);
      break;
    case CudaPDFType::CUDA_PDF_MIXTURE:
      cuda_pdf.data.mixture_pdf = *reinterpret_cast<const CudaMixturePDF *>(
          cuda_scatter_record.pdf_data);
      break;
    default:
      cuda_pdf.type = CudaPDFType::CUDA_PDF_COSINE;
      break;
    }

    cpu_scatter_record.pdf_ptr = cuda_to_cpu_pdf(cuda_pdf);
  } else {
    cpu_scatter_record.pdf_ptr = nullptr;
  }

  // Copy skip_pdf flag
  cpu_scatter_record.skip_pdf = cuda_scatter_record.skip_pdf;

  // Convert skip_pdf_ray
  cpu_scatter_record.skip_pdf_ray =
      cuda_to_cpu_ray(cuda_scatter_record.skip_pdf_ray);

  return cpu_scatter_record;
}

#endif // USE_CUDA