#ifdef USE_CUDA

#include "IntervalConversions.cuh"

// Batch conversion kernel from CPU to CUDA Interval
__global__ void batch_cpu_to_cuda_interval_kernel(const Interval *cpu_intervals,
                                                  CudaInterval *cuda_intervals,
                                                  int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    cuda_intervals[idx] = cpu_to_cuda_interval(cpu_intervals[idx]);
  }
}

// Batch conversion kernel from CUDA to CPU Interval
__global__ void
batch_cuda_to_cpu_interval_kernel(const CudaInterval *cuda_intervals,
                                  Interval *cpu_intervals, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    cpu_intervals[idx] = cuda_to_cpu_interval(cuda_intervals[idx]);
  }
}

void batch_cpu_to_cuda_interval(const Interval *cpu_intervals,
                                CudaInterval *cuda_intervals, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cpu_to_cuda_interval_kernel<<<grid_size, block_size>>>(
      cpu_intervals, cuda_intervals, count);
  cudaDeviceSynchronize();
}

void batch_cuda_to_cpu_interval(const CudaInterval *cuda_intervals,
                                Interval *cpu_intervals, int count) {
  dim3 block_size(256);
  dim3 grid_size((count + block_size.x - 1) / block_size.x);
  batch_cuda_to_cpu_interval_kernel<<<grid_size, block_size>>>(
      cuda_intervals, cpu_intervals, count);
  cudaDeviceSynchronize();
}

#endif // USE_CUDA