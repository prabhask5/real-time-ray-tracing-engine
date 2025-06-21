#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Use compile-time constants for better optimization.
#include <math_constants.h>
#define CUDA_INF INFINITY
#define CUDA_PI 3.1415926535897932385

__device__ inline double cuda_degrees_to_radians(double degrees) {
  return degrees * CUDA_PI / 180.0;
}

// Returns a random real in [0,1).
__device__ inline double cuda_random_double(curandState *state) {
  return curand_uniform_double(state);
}

// Returns a random real in [min,max).
__device__ inline double cuda_random_double(curandState *state, double min,
                                            double max) {
  return min + (max - min) * cuda_random_double(state);
}

// Optimized random integer generation.
__device__ inline int cuda_random_int(curandState *state, int min, int max) {
  // Use curand directly for integer generation when possible.
  return min + (curand(state) % (max - min + 1));
}

#endif // USE_CUDA
