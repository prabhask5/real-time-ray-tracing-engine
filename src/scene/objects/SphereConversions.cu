#ifdef USE_CUDA

#include "SphereConversions.cuh"

// Batch conversion kernel from CPU to CUDA Sphere.
__global__ void
batch_cpu_to_cuda_sphere_kernel(const Sphere **cpu_spheres,
                                const CudaMaterial *cuda_materials,
                                CudaSphere *cuda_spheres, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    if (cpu_spheres[idx] != nullptr) {
      cuda_spheres[idx] =
          cpu_to_cuda_sphere(*cpu_spheres[idx], cuda_materials[idx]);
    } else {
      // Create default sphere for null pointers.
      CudaMaterial default_material =
          cuda_make_lambertian_material(Color(0.7, 0.7, 0.7));
      cuda_spheres[idx] =
          create_cuda_sphere_static(Point3(0, 0, 0), 1.0, default_material);
    }
  }
}

void batch_cpu_to_cuda_sphere(const Sphere **cpu_spheres,
                              const CudaMaterial *cuda_materials,
                              CudaSphere *cuda_spheres, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cpu_to_cuda_sphere_kernel<<<grid_size, block_size>>>(
      cpu_spheres, cuda_materials, cuda_spheres, count);
  cudaDeviceSynchronize();
}

void batch_cuda_to_cpu_sphere(const CudaSphere *cuda_spheres,
                              std::shared_ptr<Sphere> *cpu_spheres, int count) {
  // This needs to be done on CPU since we're creating shared_ptr objects.
  for (int i = 0; i < count; i++) {
    cpu_spheres[i] = cuda_to_cpu_sphere(cuda_spheres[i]);
  }
}

#endif // USE_CUDA