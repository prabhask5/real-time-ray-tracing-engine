#pragma once

#ifdef USE_CUDA

#include "../utils/math/PDFConversions.cuh"
#include "../utils/math/Vec3Conversions.cuh"
#include "RayConversions.cuh"
#include "ScatterRecord.cuh"
#include "ScatterRecord.hpp"

// Convert CPU ScatterRecord to CUDA ScatterRecord POD struct.
inline CudaScatterRecord
cpu_to_cuda_scatter_record(const ScatterRecord &cpu_scatter_record) {
  CudaScatterRecord cuda_scatter_record;
  cuda_scatter_record.attenuation =
      cpu_to_cuda_vec3(cpu_scatter_record.attenuation);

  // Leave for now.
  cuda_scatter_record.pdf = nullptr;

  cuda_scatter_record.skip_pdf = cpu_scatter_record.skip_pdf;
  cuda_scatter_record.skip_pdf_ray =
      cpu_to_cuda_ray(cpu_scatter_record.skip_pdf_ray);
  return cuda_scatter_record;
}

#endif // USE_CUDA