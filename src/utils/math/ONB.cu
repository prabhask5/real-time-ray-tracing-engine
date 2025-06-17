#include "ONB.cuh"

#ifdef USE_CUDA

// Batch ONB operations for better GPU utilization.

__global__ void create_onb_from_normals_kernel(const Vec3 *normals, ONB *onbs,
                                               int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    onbs[idx] = make_onb(normals[idx]);
  }
}

void batch_create_onb_from_normals(const Vec3 *d_normals, ONB *d_onbs,
                                   int count) {
  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
  create_onb_from_normals_kernel<<<numBlocks, blockSize>>>(d_normals, d_onbs,
                                                           count);
  cudaDeviceSynchronize();
}

// Batch vector transformation.
__global__ void transform_vectors_kernel(const ONB *onbs,
                                         const Vec3 *local_vectors,
                                         Vec3 *world_vectors, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    world_vectors[idx] = onb_transform(onbs[idx], local_vectors[idx]);
  }
}

void batch_transform_vectors(const ONB *d_onbs, const Vec3 *d_local_vectors,
                             Vec3 *d_world_vectors, int count) {
  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
  transform_vectors_kernel<<<numBlocks, blockSize>>>(d_onbs, d_local_vectors,
                                                     d_world_vectors, count);
  cudaDeviceSynchronize();
}

#endif // USE_CUDA