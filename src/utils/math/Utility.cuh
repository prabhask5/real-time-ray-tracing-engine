#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ constexpr double INF = CUDART_INF;
__device__ constexpr double PI = 3.1415926535897932385;

__device__ inline double degrees_to_radians(double degrees) {
  return degrees * PI / 180.0;
}

// Returns a random real in [0,1).
__device__ inline double random_double(curandState *state) {
  return curand_uniform_double(state);
}

// Returns a random real in [min,max).
__device__ inline double random_double(curandState *state, double min,
                                       double max) {
  return min + (max - min) * random_double(state);
}

// Returns a random integer in [min,max].
__device__ inline int random_int(curandState *state, int min, int max) {
  return static_cast<int>(random_double(state, min, max + 1));
}

#endif // USE_CUDA
