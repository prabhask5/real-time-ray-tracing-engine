#pragma once

#ifdef USE_CUDA

#include "CameraKernels.cuh"
#include <cuda_runtime.h>

// Wrapper functions that can be called from .cpp files.
// These hide the CUDA kernel launch syntax.

void cuda_init_rand_states_wrapper(curandState *d_rand_states, int width,
                                   int height, unsigned long seed,
                                   dim3 grid_size, dim3 block_size);

void cuda_dynamic_render_tile_wrapper(
    CudaColor *accumulation, int image_width, int image_height, int start_r,
    int end_r, int start_c, int end_c, int s_i, int s_j, int sqrt_spp,
    int max_depth, CudaPoint3 center, CudaPoint3 pixel00_loc,
    CudaVec3 pixel_delta_u, CudaVec3 pixel_delta_v, CudaVec3 u, CudaVec3 v,
    CudaVec3 w, CudaVec3 defocus_disk_u, CudaVec3 defocus_disk_v,
    double defocus_angle, CudaColor background, CudaHittable world,
    CudaHittable lights, curandState *rand_states, dim3 grid_size,
    dim3 block_size);

void cuda_static_render_wrapper(
    CudaColor *pixel_colors, int image_width, int image_height, int start_row,
    int end_row, int sqrt_spp, int max_depth, CudaPoint3 center,
    CudaPoint3 pixel00_loc, CudaVec3 pixel_delta_u, CudaVec3 pixel_delta_v,
    CudaVec3 u, CudaVec3 v, CudaVec3 w, CudaVec3 defocus_disk_u,
    CudaVec3 defocus_disk_v, double defocus_angle, double pixel_samples_scale,
    CudaColor background, CudaHittable world, CudaHittable lights,
    curandState *rand_states, dim3 grid_size, dim3 block_size);

#endif // USE_CUDA