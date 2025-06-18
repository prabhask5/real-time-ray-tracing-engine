#pragma once

#ifdef USE_CUDA

#include "../core/Vec3Types.cuh"
#include "math/Interval.cuh"
#include <cuda_runtime.h>

// Applies gamma correction with gamma = 2.0.
// Converts from linear space (physical light) to gamma space (display
// brightness).
__device__ inline double cuda_linear_to_gamma(double linear_component) {
  if (linear_component > 0)
    return sqrt(linear_component);
  return 0.0;
}

// Optimized color value to byte conversion with gamma correction.
__device__ inline unsigned char cuda_to_byte(double color_value) {
  double x = cuda_linear_to_gamma(color_value);
  // Manual clamp is faster than creating Interval object.
  x = fmax(0.000, fmin(0.999, x));
  return static_cast<unsigned char>(256 * x);
}

// Device-side conversion of a Color to RGB byte triplet.
__device__ inline void cuda_color_to_bytes(const CudaColor &pixel_color,
                                           unsigned char &r, unsigned char &g,
                                           unsigned char &b) {
  r = cuda_to_byte(pixel_color.x);
  g = cuda_to_byte(pixel_color.y);
  b = cuda_to_byte(pixel_color.z);
}

// GPU batch processing functions (implemented in .cu file).
void cuda_convert_colors_to_rgb(const CudaColor *d_colors,
                                unsigned char *d_rgb_data, int width,
                                int height, int samples_per_pixel);
void cuda_accumulate_colors(CudaColor *d_accumulation_buffer,
                            const CudaColor *d_new_colors, int count,
                            int sample_number);

#endif // USE_CUDA
