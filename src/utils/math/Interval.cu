#include "Interval.cuh"

#ifdef USE_CUDA

// Batch interval operations for better GPU utilization.

__global__ void cuda_clamp_values_kernel(double *values,
                                         const CudaInterval *intervals,
                                         int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    values[idx] = intervals[idx].clamp(values[idx]);
  }
}

void cuda_batch_clamp_values(double *d_values, const CudaInterval *d_intervals,
                             int count) {
  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
  cuda_clamp_values_kernel<<<numBlocks, blockSize>>>(d_values, d_intervals,
                                                     count);
  cudaDeviceSynchronize();
}

// Optimized interval intersection.
__global__ void cuda_intersect_intervals_kernel(const CudaInterval *intervals1,
                                                const CudaInterval *intervals2,
                                                CudaInterval *result,
                                                int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    double min_val = fmax(intervals1[idx].min, intervals2[idx].min);
    double max_val = fmin(intervals1[idx].max, intervals2[idx].max);
    result[idx] = CudaInterval(min_val, max_val);
  }
}

void cuda_batch_intersect_intervals(const CudaInterval *d_intervals1,
                                    const CudaInterval *d_intervals2,
                                    CudaInterval *d_result, int count) {
  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
  cuda_intersect_intervals_kernel<<<numBlocks, blockSize>>>(
      d_intervals1, d_intervals2, d_result, count);
  cudaDeviceSynchronize();
}

#endif // USE_CUDA