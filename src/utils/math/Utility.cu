#include "Utility.cuh"

#ifdef USE_CUDA

// Optimized batch random number generation kernels.

__global__ void cuda_init_random_states_kernel(curandState *states, int count,
                                               unsigned long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    curand_init(seed, idx, 0, &states[idx]);
  }
}

// Host function to initialize random states efficiently.
void cuda_init_random_states_batch(curandState **d_states, int count,
                                   unsigned long seed) {
  cudaMalloc(d_states, count * sizeof(curandState));

  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
  cuda_init_random_states_kernel<<<numBlocks, blockSize>>>(*d_states, count,
                                                           seed);
  cudaDeviceSynchronize();
}

void cuda_free_random_states_batch(curandState *d_states) {
  if (d_states) {
    cudaFree(d_states);
  }
}

// Optimized batch angle conversion
__global__ void cuda_degrees_to_radians_kernel(const double *degrees,
                                               double *radians, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    radians[idx] = degrees_to_radians(degrees[idx]);
  }
}

void cuda_batch_degrees_to_radians(const double *d_degrees, double *d_radians,
                                   int count) {
  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
  cuda_degrees_to_radians_kernel<<<numBlocks, blockSize>>>(d_degrees, d_radians,
                                                           count);
  cudaDeviceSynchronize();
}

#endif // USE_CUDA