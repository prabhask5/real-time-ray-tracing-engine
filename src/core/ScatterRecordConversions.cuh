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
    CudaPDF cuda_pdf = cpu_to_cuda_pdf(*cpu_scatter_record.pdf_ptr);
    cuda_scatter_record.pdf_pointer = &cuda_pdf;
  }

  // Copy skip_pdf flag.
  cuda_scatter_record.skip_pdf = cpu_scatter_record.skip_pdf;

  // Convert skip_pdf_ray.
  cuda_scatter_record.skip_pdf_ray =
      cpu_to_cuda_ray(cpu_scatter_record.skip_pdf_ray);

  return cuda_scatter_record;
}

// Convert CUDA ScatterRecord to CPU ScatterRecord.
inline ScatterRecord
cuda_to_cpu_scatter_record(const CudaScatterRecord &cuda_scatter_record) {
  ScatterRecord cpu_scatter_record;

  // Convert attenuation color.
  cpu_scatter_record.attenuation =
      cuda_to_cpu_vec3(cuda_scatter_record.attenuation);

  // Handle PDF conversion - CUDA uses type enum + raw pointer, CPU uses
  // shared_ptr.
  if (cuda_scatter_record.pdf_pointer != nullptr) {
    cpu_scatter_record.pdf_ptr =
        cuda_to_cpu_pdf(*cuda_scatter_record.pdf_pointer);
  } else {
    cpu_scatter_record.pdf_ptr = nullptr;
  }

  // Copy skip_pdf flag.

  cpu_scatter_record.skip_pdf = cuda_scatter_record.skip_pdf;

  // Convert skip_pdf_ray.

  cpu_scatter_record.skip_pdf_ray =
      cuda_to_cpu_ray(cuda_scatter_record.skip_pdf_ray);

  return cpu_scatter_record;
}

#endif // USE_CUDA