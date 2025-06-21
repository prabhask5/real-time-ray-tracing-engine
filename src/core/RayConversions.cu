#ifdef USE_CUDA

#include "RayConversions.cuh"

// Batch conversion kernel from CPU to CUDA Ray.

__global__ void batch_cpu_to_cuda_ray_kernel(const Ray *cpu_rays,
                                             CudaRay *cuda_rays, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    cuda_rays[idx] = cpu_to_cuda_ray(cpu_rays[idx]);
  }
}

// Batch conversion kernel from CUDA to CPU Ray.

__global__ void batch_cuda_to_cpu_ray_kernel(const CudaRay *cuda_rays,
                                             Ray *cpu_rays, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    cpu_rays[idx] = cuda_to_cpu_ray(cuda_rays[idx]);
  }
}

void batch_cpu_to_cuda_ray(const Ray *cpu_rays, CudaRay *cuda_rays, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cpu_to_cuda_ray_kernel<<<grid_size, block_size>>>(cpu_rays, cuda_rays,
                                                          count);
  cudaDeviceSynchronize();
}

void batch_cuda_to_cpu_ray(const CudaRay *cuda_rays, Ray *cpu_rays, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cuda_to_cpu_ray_kernel<<<grid_size, block_size>>>(cuda_rays, cpu_rays,
                                                          count);
  cudaDeviceSynchronize();
}

#endif // USE_CUDA