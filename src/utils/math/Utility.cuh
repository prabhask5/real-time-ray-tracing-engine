#pragma once

#ifdef USE_CUDA

#include <curand_kernel.h>
#include <cuda_runtime.h>

__device__ constexpr double PI = 3.1415926535897932385;
__device__ constexpr double INF = CUDART_INF;

__device__ double degrees_to_radians(double degrees);

// Returns a random real in [0,1).
__device__ double random_double(curandState* state);

// Returns a random real in [min,max).
__device__ double random_double(curandState* state, double min, double max);

// Returns a random integer in [min,max].
__device__ int random_int(curandState* state, int min, int max);

#endif // USE_CUDA