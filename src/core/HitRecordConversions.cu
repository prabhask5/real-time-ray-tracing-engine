#ifdef USE_CUDA

#include "HitRecordConversions.cuh"

// Batch conversion kernel from CPU to CUDA HitRecord.
__global__ void
batch_cpu_to_cuda_hit_record_kernel(const HitRecord *cpu_records,
                                    CudaHitRecord *cuda_records, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    cuda_records[idx] = cpu_to_cuda_hit_record(cpu_records[idx]);
  }
}

void batch_cpu_to_cuda_hit_record(const HitRecord *cpu_records,
                                  CudaHitRecord *cuda_records, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cpu_to_cuda_hit_record_kernel<<<grid_size, block_size>>>(
      cpu_records, cuda_records, count);
  cudaDeviceSynchronize();
}

void batch_cuda_to_cpu_hit_record(const CudaHitRecord *cuda_records,
                                  HitRecord *cpu_records, int count) {
  // This needs to be done on CPU since we're creating shared_ptr objects.
  for (int i = 0; i < count; i++) {
    cpu_records[i] = cuda_to_cpu_hit_record(cuda_records[i]);
  }
}

#endif // USE_CUDA