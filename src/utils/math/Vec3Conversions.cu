#ifdef USE_CUDA

#include "Vec3Conversions.cuh"

// Batch conversion kernel from CPU to CUDA Vec3
__global__ void batch_cpu_to_cuda_vec3_kernel(const Vec3 *cpu_vecs,
                                              CudaVec3 *cuda_vecs, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    cuda_vecs[idx] = cpu_to_cuda_vec3(cpu_vecs[idx]);
  }
}

// Batch conversion kernel from CUDA to CPU Vec3
__global__ void batch_cuda_to_cpu_vec3_kernel(const CudaVec3 *cuda_vecs,
                                              Vec3 *cpu_vecs, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    cpu_vecs[idx] = cuda_to_cpu_vec3(cuda_vecs[idx]);
  }
}

void batch_cpu_to_cuda_vec3(const Vec3 *cpu_vecs, CudaVec3 *cuda_vecs,
                            int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cpu_to_cuda_vec3_kernel<<<grid_size, block_size>>>(cpu_vecs, cuda_vecs,
                                                           count);
  cudaDeviceSynchronize();
}

void batch_cuda_to_cpu_vec3(const CudaVec3 *cuda_vecs, Vec3 *cpu_vecs,
                            int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cuda_to_cpu_vec3_kernel<<<grid_size, block_size>>>(cuda_vecs, cpu_vecs,
                                                           count);
  cudaDeviceSynchronize();
}

#endif // USE_CUDA