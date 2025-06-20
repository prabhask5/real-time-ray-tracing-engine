#pragma once

#ifdef USE_CUDA

#include "PerlinNoise.cuh"
#include "PerlinNoise.hpp"
#include "Vec3Conversions.cuh"

// Convert CPU PerlinNoise to CUDA PerlinNoise.
inline CudaPerlinNoise cpu_to_cuda_perlin_noise(const PerlinNoise &cpu_perlin) {
  CudaVec3 rand_vec[CUDA_PERLIN_POINT_COUNT];
  const Vec3 *cpu_rand_vec = cpu_perlin.rand_vec();
  for (int i = 0; i < CUDA_PERLIN_POINT_COUNT; ++i)
    rand_vec[i] = cpu_to_cuda_vec3(cpu_rand_vec[i]);

  return CudaPerlinNoise(rand_vec, cpu_perlin.perm_x(), cpu_perlin.perm_y(),
                         cpu_perlin.perm_z());
}

// Convert CUDA PerlinNoise to CPU PerlinNoise.
inline PerlinNoise
cuda_to_cpu_perlin_noise(const CudaPerlinNoise &cuda_perlin) {
  Vec3 rand_vec[CUDA_PERLIN_POINT_COUNT];
  for (int i = 0; i < CUDA_PERLIN_POINT_COUNT; ++i)
    rand_vec[i] = cuda_to_cpu_vec3(cuda_perlin.rand_vec[i]);

  return PerlinNoise(rand_vec, cuda_perlin.perm_x, cuda_perlin.perm_y,
                     cuda_perlin.perm_z);
}

// Batch conversion functions for performance.
void batch_cpu_to_cuda_perlin_noise(const PerlinNoise *cpu_perlins,
                                    CudaPerlinNoise *cuda_perlins, int count);
void batch_cuda_to_cpu_perlin_noise(const CudaPerlinNoise *cuda_perlins,
                                    PerlinNoise *cpu_perlins, int count);

#endif // USE_CUDA