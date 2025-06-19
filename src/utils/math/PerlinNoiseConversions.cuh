#pragma once

#ifdef USE_CUDA

#include "PerlinNoise.cuh"
#include "PerlinNoise.hpp"
#include "Vec3Conversions.cuh"

// Convert CPU PerlinNoise to CUDA PerlinNoise.
__host__ inline CudaPerlinNoise
cpu_to_cuda_perlin_noise(const PerlinNoise &cpu_perlin) {
  return CudaPerlinNoise(cpu_perlin.rand_vec(), cpu_perlin.perm_x(),
                         cpu_perlin.perm_y(), cpu_perlin.perm_z());
}

// Convert CUDA PerlinNoise to CPU PerlinNoise.
__host__ inline PerlinNoise
cuda_to_cpu_perlin_noise(const CudaPerlinNoise &cuda_perlin) {
  return PerlinNoise(cuda_perlin.rand_vec, cuda_perlin.perm_x,
                     cuda_perlin.perm_y, cuda_perlin.perm_z);
}

// Batch conversion functions for performance.
__host__ void batch_cpu_to_cuda_perlin_noise(const PerlinNoise *cpu_perlins,
                                             CudaPerlinNoise *cuda_perlins,
                                             int count);
__host__ void
batch_cuda_to_cpu_perlin_noise(const CudaPerlinNoise *cuda_perlins,
                               PerlinNoise *cpu_perlins, int count);

#endif // USE_CUDA