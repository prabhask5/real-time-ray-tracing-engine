#ifdef USE_CUDA

#include "MaterialConversions.cuh"

// Batch conversion kernel from CPU to CUDA Material.
__global__ void
batch_cpu_to_cuda_material_kernel(const Material **cpu_materials,
                                  CudaMaterial *cuda_materials, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    if (cpu_materials[idx] != nullptr) {
      cuda_materials[idx] = cpu_to_cuda_material(*cpu_materials[idx]);
    } else {
      // Create default Lambertian material for null pointers.
      cuda_materials[idx] = cuda_make_lambertian_material(Color(0.7, 0.7, 0.7));
    }
  }
}

void batch_cpu_to_cuda_material(const Material **cpu_materials,
                                CudaMaterial *cuda_materials, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cpu_to_cuda_material_kernel<<<grid_size, block_size>>>(
      cpu_materials, cuda_materials, count);
  cudaDeviceSynchronize();
}

void batch_cuda_to_cpu_material(const CudaMaterial *cuda_materials,
                                MaterialPtr *cpu_materials, int count) {
  // This needs to be done on CPU since we're creating shared_ptr objects.
  for (int i = 0; i < count; i++) {
    cpu_materials[i] = cuda_to_cpu_material(cuda_materials[i]);
  }
}

#endif // USE_CUDA