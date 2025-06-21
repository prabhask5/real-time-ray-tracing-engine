#include "ONB.cuh"

#ifdef USE_CUDA

// Batch ONB operations for better GPU utilization.

__global__ void cuda_create_onb_from_normals_kernel(const CudaVec3 *normals,
                                                    CudaONB *onbs, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    onbs[idx] = CudaONB(normals[idx]);
  }
}

void cuda_batch_create_onb_from_normals(const CudaVec3 *d_normals,
                                        CudaONB *d_onbs, int count) {
  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
  cuda_create_onb_from_normals_kernel<<<numBlocks, blockSize>>>(d_normals,
                                                                d_onbs, count);
  cudaDeviceSynchronize();
}

// Batch vector transformation.
__global__ void cuda_transform_vectors_kernel(const CudaONB *onbs,
                                              const CudaVec3 *local_vectors,
                                              CudaVec3 *world_vectors,
                                              int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    world_vectors[idx] = onbs[idx].transform(local_vectors[idx]);
  }
}

void cuda_batch_transform_vectors(const CudaONB *d_onbs,
                                  const CudaVec3 *d_local_vectors,
                                  CudaVec3 *d_world_vectors, int count) {
  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
  cuda_transform_vectors_kernel<<<numBlocks, blockSize>>>(
      d_onbs, d_local_vectors, d_world_vectors, count);
  cudaDeviceSynchronize();
}

#endif // USE_CUDA