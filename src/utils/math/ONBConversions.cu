#ifdef USE_CUDA

#include "ONBConversions.cuh"

// Batch conversion kernel from CPU to CUDA ONB
__global__ void batch_cpu_to_cuda_onb_kernel(const ONB *cpu_onbs,
                                             CudaONB *cuda_onbs, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    cuda_onbs[idx] = cpu_to_cuda_onb(cpu_onbs[idx]);
  }
}

// Batch conversion kernel from CUDA to CPU ONB
__global__ void batch_cuda_to_cpu_onb_kernel(const CudaONB *cuda_onbs,
                                             ONB *cpu_onbs, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    cpu_onbs[idx] = cuda_to_cpu_onb(cuda_onbs[idx]);
  }
}

void batch_cpu_to_cuda_onb(const ONB *cpu_onbs, CudaONB *cuda_onbs, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cpu_to_cuda_onb_kernel<<<grid_size, block_size>>>(cpu_onbs, cuda_onbs,
                                                          count);
  cudaDeviceSynchronize();
}

void batch_cuda_to_cpu_onb(const CudaONB *cuda_onbs, ONB *cpu_onbs, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cuda_to_cpu_onb_kernel<<<grid_size, block_size>>>(cuda_onbs, cpu_onbs,
                                                          count);
  cudaDeviceSynchronize();
}

#endif // USE_CUDA