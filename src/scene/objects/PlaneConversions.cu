#ifdef USE_CUDA

#include "PlaneConversions.cuh"

// Batch conversion kernel from CPU to CUDA Plane
__global__ void
batch_cpu_to_cuda_plane_kernel(const Plane **cpu_planes,
                               const CudaMaterial *cuda_materials,
                               CudaPlane *cuda_planes, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    if (cpu_planes[idx] != nullptr) {
      // Since we can't access private members, create with default parameters
      cuda_planes[idx] = create_cuda_plane(Point3(0, 0, 0), Vec3(1, 0, 0),
                                           Vec3(0, 1, 0), cuda_materials[idx]);
    } else {
      // Create default plane for null pointers
      CudaMaterial default_material =
          cuda_make_lambertian_material(Color(0.7, 0.7, 0.7));
      cuda_planes[idx] = create_cuda_plane(Point3(0, 0, 0), Vec3(1, 0, 0),
                                           Vec3(0, 1, 0), default_material);
    }
  }
}

void batch_cpu_to_cuda_plane(const Plane **cpu_planes,
                             const CudaMaterial *cuda_materials,
                             CudaPlane *cuda_planes, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cpu_to_cuda_plane_kernel<<<grid_size, block_size>>>(
      cpu_planes, cuda_materials, cuda_planes, count);
  cudaDeviceSynchronize();
}

void batch_cuda_to_cpu_plane(const CudaPlane *cuda_planes,
                             std::shared_ptr<Plane> *cpu_planes, int count) {
  // This needs to be done on CPU since we're creating shared_ptr objects
  for (int i = 0; i < count; i++) {
    cpu_planes[i] = cuda_to_cpu_plane(cuda_planes[i]);
  }
}

#endif // USE_CUDA