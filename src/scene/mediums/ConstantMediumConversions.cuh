#pragma once

#ifdef USE_CUDA

#include "../materials/MaterialConversions.cuh"
#include "../textures/SolidColorTexture.hpp"
#include "../textures/TextureConversions.cuh"
#include "ConstantMedium.cuh"
#include "ConstantMedium.hpp"

// Forward declaration for hittable conversion.
struct CudaHittable;
CudaHittable cpu_to_cuda_hittable(const Hittable &cpu_hittable);

// Convert CPU ConstantMedium to CUDA ConstantMedium.
inline CudaConstantMedium
cpu_to_cuda_constant_medium(const ConstantMedium &cpu_constant_medium) {
  HittablePtr cpu_boundary = cpu_constant_medium.get_boundary();
  double density = cpu_constant_medium.get_density();
  MaterialPtr cpu_phase_function = cpu_constant_medium.get_phase_function();

  CudaHittable *cuda_boundary = new CudaHittable();
  *cuda_boundary = cpu_to_cuda_hittable(*cpu_boundary);

  CudaMaterial *cuda_phase_function = new CudaMaterial();
  *cuda_phase_function = cpu_to_cuda_material(*cpu_phase_function);

  return cuda_make_constant_medium(cuda_boundary, density, cuda_phase_function);
}

#endif // USE_CUDA