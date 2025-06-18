#include "Vec3Utility.cuh"

#ifdef USE_CUDA

// Additional optimized utility functions that benefit from being in .cu file.

// Batch random vector generation for better GPU utilization.
__global__ void cuda_generate_random_vectors_kernel(CudaVec3 *vectors,
                                                    int count,
                                                    curandState *states) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    vectors[idx] = cuda_random_unit_vector(&states[idx]);
  }
}

// Host function to generate many random vectors efficiently on GPU.
void cuda_generate_random_vectors(CudaVec3 *d_vectors, int count,
                                  curandState *d_states) {
  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
  cuda_generate_random_vectors_kernel<<<numBlocks, blockSize>>>(
      d_vectors, count, d_states);
  cudaDeviceSynchronize();
}

// Optimized vector operations for arrays.
__global__ void cuda_normalize_vectors_kernel(CudaVec3 *vectors, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    vectors[idx] = cuda_unit_vector(vectors[idx]);
  }
}

void cuda_normalize_vectors(CudaVec3 *d_vectors, int count) {
  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
  cuda_normalize_vectors_kernel<<<numBlocks, blockSize>>>(d_vectors, count);
  cudaDeviceSynchronize();
}

#endif // USE_CUDA