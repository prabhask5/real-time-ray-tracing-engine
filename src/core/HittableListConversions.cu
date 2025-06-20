#ifdef USE_CUDA

#include "HittableListConversions.cuh"

// Batch conversion kernel from CPU to CUDA HittableList.
__global__ void batch_cpu_to_cuda_hittable_list_kernel(
    const HittableList *cpu_lists, CudaHittableList *cuda_lists, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    // For GPU kernel, we can only do basic initialization.
    // Complex conversion with dynamic memory allocation should be done on host.
    cuda_lists[idx] = create_empty_cuda_hittable_list();
  }
}

void batch_cpu_to_cuda_hittable_list(const HittableList *cpu_lists,
                                     CudaHittableList *cuda_lists, int count) {
  // This needs to be done on CPU due to complex object conversion requirements.
  for (int i = 0; i < count; i++) {
    CudaHittable temp_buffer[MAX_HITTABLE_CONVERSION_COUNT];
    cuda_lists[i] = cpu_to_cuda_hittable_list(cpu_lists[i], temp_buffer,
                                              MAX_HITTABLE_CONVERSION_COUNT);
  }
}

void batch_cuda_to_cpu_hittable_list(const CudaHittableList *cuda_lists,
                                     HittableList *cpu_lists, int count) {
  // This needs to be done on CPU since we're creating shared_ptr objects.
  for (int i = 0; i < count; i++) {
    cpu_lists[i] = cuda_to_cpu_hittable_list(cuda_lists[i]);
  }
}

#endif // USE_CUDA