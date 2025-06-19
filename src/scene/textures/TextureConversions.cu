#ifdef USE_CUDA

#include "TextureConversions.cuh"

// Batch conversion kernel from CPU to CUDA Texture.
__global__ void batch_cpu_to_cuda_texture_kernel(const Texture **cpu_textures,
                                                 CudaTexture *cuda_textures,
                                                 int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    if (cpu_textures[idx] != nullptr) {
      cuda_textures[idx] = cpu_to_cuda_texture(*cpu_textures[idx]);
    } else {
      // Create default solid texture for null pointers.
      cuda_textures[idx] = cuda_make_solid_texture(Color(0.7, 0.7, 0.7));
    }
  }
}

void batch_cpu_to_cuda_texture(const Texture **cpu_textures,
                               CudaTexture *cuda_textures, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cpu_to_cuda_texture_kernel<<<grid_size, block_size>>>(
      cpu_textures, cuda_textures, count);
  cudaDeviceSynchronize();
}

void batch_cuda_to_cpu_texture(const CudaTexture *cuda_textures,
                               TexturePtr *cpu_textures, int count) {
  // This needs to be done on CPU since we're creating shared_ptr objects.
  for (int i = 0; i < count; i++) {
    cpu_textures[i] = cuda_to_cpu_texture(cuda_textures[i]);
  }
}

#endif // USE_CUDA