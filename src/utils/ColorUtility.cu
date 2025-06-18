#include "ColorUtility.cuh"

#ifdef USE_CUDA

// Optimized batch color processing kernels for better GPU utilization.

__global__ void cuda_colors_to_bytes_kernel(const CudaColor *colors,
                                            unsigned char *rgb_data, int width,
                                            int height, int samples_per_pixel) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int pixel_index = y * width + x;
  CudaColor pixel_color = colors[pixel_index];

  // Apply sample averaging if needed.
  if (samples_per_pixel > 1) {
    double scale = 1.0 / samples_per_pixel;
    pixel_color = scale * pixel_color;
  }

  // Convert to bytes with gamma correction.
  int rgb_index = pixel_index * 3;
  cuda_color_to_bytes(pixel_color, rgb_data[rgb_index], rgb_data[rgb_index + 1],
                      rgb_data[rgb_index + 2]);
}

// Host function for efficient batch color to RGB conversion.
void cuda_convert_colors_to_rgb(const CudaColor *d_colors,
                                unsigned char *d_rgb_data, int width,
                                int height, int samples_per_pixel) {
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  cuda_colors_to_bytes_kernel<<<gridSize, blockSize>>>(
      d_colors, d_rgb_data, width, height, samples_per_pixel);
  cudaDeviceSynchronize();
}

// Optimized color accumulation kernel.
__global__ void cuda_accumulate_colors_kernel(CudaColor *accumulation_buffer,
                                              const CudaColor *new_colors,
                                              int count, int sample_number) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    // Progressive averaging for numerical stability.
    double inv_samples = 1.0 / sample_number;
    accumulation_buffer[idx] =
        (accumulation_buffer[idx] * (sample_number - 1) + new_colors[idx]) *
        inv_samples;
  }
}

void cuda_accumulate_colors(CudaColor *d_accumulation_buffer,
                            const CudaColor *d_new_colors, int count,
                            int sample_number) {
  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
  cuda_accumulate_colors_kernel<<<numBlocks, blockSize>>>(
      d_accumulation_buffer, d_new_colors, count, sample_number);
  cudaDeviceSynchronize();
}

#endif // USE_CUDA