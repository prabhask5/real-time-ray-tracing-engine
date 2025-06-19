#ifdef USE_CUDA

#include "PerlinNoiseConversions.cuh"

// Batch conversion kernel from CPU to CUDA PerlinNoise.
__global__ void batch_cpu_to_cuda_perlin_noise_kernel(
    const PerlinNoise *cpu_perlins, CudaPerlinNoise *cuda_perlins, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    cuda_perlins[idx] = cpu_to_cuda_perlin_noise(cpu_perlins[idx]);
  }
}

// Batch conversion kernel from CUDA to CPU PerlinNoise.
__global__ void
batch_cuda_to_cpu_perlin_noise_kernel(const CudaPerlinNoise *cuda_perlins,
                                      PerlinNoise *cpu_perlins, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    cpu_perlins[idx] = cuda_to_cpu_perlin_noise(cuda_perlins[idx]);
  }
}

void batch_cpu_to_cuda_perlin_noise(const PerlinNoise *cpu_perlins,
                                    CudaPerlinNoise *cuda_perlins, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cpu_to_cuda_perlin_noise_kernel<<<grid_size, block_size>>>(
      cpu_perlins, cuda_perlins, count);
  cudaDeviceSynchronize();
}

void batch_cuda_to_cpu_perlin_noise(const CudaPerlinNoise *cuda_perlins,
                                    PerlinNoise *cpu_perlins, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cuda_to_cpu_perlin_noise_kernel<<<grid_size, block_size>>>(
      cuda_perlins, cpu_perlins, count);
  cudaDeviceSynchronize();
}

#endif // USE_CUDA