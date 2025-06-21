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
HittablePtr cuda_to_cpu_hittable(const CudaHittable &cuda_hittable);

// Convert CPU ConstantMedium to CUDA ConstantMedium.
inline CudaConstantMedium
cpu_to_cuda_constant_medium(const ConstantMedium &cpu_constant_medium) {
  // Create CUDA constant medium with proper constructor.
  HittablePtr cpu_boundary = cpu_constant_medium.get_boundary();
  double density = cpu_constant_medium.get_density();
  MaterialPtr cpu_phase_function = cpu_constant_medium.get_phase_function();

  // Convert components.
  CudaHittable cuda_boundary = cpu_to_cuda_hittable(*cpu_boundary);
  CudaMaterial cuda_phase_function = cpu_to_cuda_material(*cpu_phase_function);

  CudaConstantMedium cuda_constant_medium(&cuda_boundary, density,
                                          &cuda_phase_function);
  return cuda_constant_medium;
}

// Convert CUDA ConstantMedium to CPU ConstantMedium.
inline ConstantMedium
cuda_to_cpu_constant_medium(const CudaConstantMedium &cuda_constant_medium) {
  // Convert boundary back to CPU format.
  HittablePtr cpu_boundary =
      cuda_to_cpu_hittable(*cuda_constant_medium.boundary);

  // Calculate density from neg_inv_density.
  double density = -1.0 / cuda_constant_medium.neg_inv_density;

  MaterialPtr cpu_phase_texture =
      cuda_to_cpu_material(*cuda_constant_medium.phase_function);

  // Create CPU ConstantMedium.
  return ConstantMedium(cpu_boundary, density, cpu_phase_texture);
}

// Memory management for constant medium objects.
inline void
cleanup_cuda_constant_medium(CudaConstantMedium &cuda_constant_medium) {
  if (cuda_constant_medium.boundary) {
    delete cuda_constant_medium.boundary;
    cuda_constant_medium.boundary = nullptr;
  }
}

#endif // USE_CUDA