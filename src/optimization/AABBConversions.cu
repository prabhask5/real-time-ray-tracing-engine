#ifdef USE_CUDA

#include "AABBConversions.cuh"

// Batch conversion kernel from CPU to CUDA AABB
__global__ void batch_cpu_to_cuda_aabb_kernel(const AABB *cpu_aabbs,
                                              CudaAABB *cuda_aabbs, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    cuda_aabbs[idx] = cpu_to_cuda_aabb(cpu_aabbs[idx]);
  }
}

// Batch conversion kernel from CUDA to CPU AABB
__global__ void batch_cuda_to_cpu_aabb_kernel(const CudaAABB *cuda_aabbs,
                                              AABB *cpu_aabbs, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    cpu_aabbs[idx] = cuda_to_cpu_aabb(cuda_aabbs[idx]);
  }
}

__host__ void batch_cpu_to_cuda_aabb(const AABB *cpu_aabbs,
                                     CudaAABB *cuda_aabbs, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cpu_to_cuda_aabb_kernel<<<grid_size, block_size>>>(cpu_aabbs,
                                                           cuda_aabbs, count);
  cudaDeviceSynchronize();
}

__host__ void batch_cuda_to_cpu_aabb(const CudaAABB *cuda_aabbs,
                                     AABB *cpu_aabbs, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cuda_to_cpu_aabb_kernel<<<grid_size, block_size>>>(cuda_aabbs,
                                                           cpu_aabbs, count);
  cudaDeviceSynchronize();
}

#endif // USE_CUDA