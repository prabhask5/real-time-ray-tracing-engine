#pragma once

#ifdef USE_CUDA

#include "PerlinNoise.cuh"
#include "PerlinNoise.hpp"
#include "Vec3Conversions.cuh"

// Convert CPU PerlinNoise to CUDA PerlinNoise POD struct.
inline CudaPerlinNoise cpu_to_cuda_perlin_noise(const PerlinNoise &cpu_perlin) {
  CudaVec3 rand_vec[CUDA_PERLIN_POINT_COUNT];
  const Vec3 *cpu_rand_vec = cpu_perlin.rand_vec();
  for (int i = 0; i < CUDA_PERLIN_POINT_COUNT; ++i)
    rand_vec[i] = cpu_to_cuda_vec3(cpu_rand_vec[i]);

  return cuda_make_perlin_noise(rand_vec, cpu_perlin.perm_x(),
                                cpu_perlin.perm_y(), cpu_perlin.perm_z());
}

#endif // USE_CUDA