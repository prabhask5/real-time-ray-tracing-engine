#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Use compile-time constants for better optimization.
static constexpr double CUDA_INF = CUDART_INF;
static constexpr double CUDA_PI = 3.1415926535897932385;

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

// GPU batch processing functions (implemented in .cu file).
void cuda_init_random_states_batch(curandState **d_states, int count,
                                   unsigned long seed);
void cuda_free_random_states_batch(curandState *d_states);
void cuda_batch_degrees_to_radians(const double *d_degrees, double *d_radians,
                                   int count);

#endif // USE_CUDA
