#pragma once

#ifdef USE_CUDA

#include "../../optimization/BVHNode.cuh"
#include "../../utils/ColorUtility.cuh"
#include "../../utils/math/Utility.cuh"
#include "../../utils/math/Vec3.cuh"
#include "../../utils/math/Vec3Utility.cuh"
#include "../Hittable.cuh"
#include "../Ray.cuh"
#include "../Vec3Types.cuh"
#include <curand_kernel.h>

// CUDA kernel for dynamic camera tile rendering.

__global__ void dynamic_render_tile_kernel(
    CudaColor *accumulation, int image_width, int image_height, int start_r,
    int end_r, int start_c, int end_c, int s_i, int s_j, int sqrt_spp,
    int max_depth, CudaPoint3 center, CudaPoint3 pixel00_loc,
    CudaVec3 pixel_delta_u, CudaVec3 pixel_delta_v, CudaVec3 u, CudaVec3 v,
    CudaVec3 w, CudaVec3 defocus_disk_u, CudaVec3 defocus_disk_v,
    double defocus_angle, CudaColor background, const CudaHittable *world,
    const CudaHittable *lights, curandState *rand_states);

// CUDA kernel for static camera rendering.

__global__ void static_render_kernel(
    CudaColor *pixel_colors, int image_width, int image_height, int start_row,
    int end_row, int sqrt_spp, int max_depth, CudaPoint3 center,
    CudaPoint3 pixel00_loc, CudaVec3 pixel_delta_u, CudaVec3 pixel_delta_v,
    CudaVec3 u, CudaVec3 v, CudaVec3 w, CudaVec3 defocus_disk_u,
    CudaVec3 defocus_disk_v, double defocus_angle, double pixel_samples_scale,
    CudaColor background, const CudaHittable *world, const CudaHittable *lights,
    curandState *rand_states);

// CUDA kernel for initializing random states.

__global__ void init_rand_states(curandState *rand_states, int width,
                                 int height, unsigned long seed);

// Device functions for camera operations.

__device__ CudaRay get_ray_cuda(int i, int j, int s_i, int s_j, int sqrt_spp,
                                CudaPoint3 center, CudaPoint3 pixel00_loc,
                                CudaVec3 pixel_delta_u, CudaVec3 pixel_delta_v,
                                CudaVec3 defocus_disk_u,
                                CudaVec3 defocus_disk_v, double defocus_angle,
                                curandState *state);

__device__ CudaVec3 sample_square_stratified_cuda(int s_i, int s_j,
                                                  int sqrt_spp,
                                                  curandState *state);

__device__ CudaVec3 sample_square_cuda(curandState *state);

__device__ CudaVec3 sample_disk_cuda(double radius, curandState *state);

__device__ CudaPoint3 defocus_disk_sample_cuda(CudaVec3 defocus_disk_u,
                                               CudaVec3 defocus_disk_v,
                                               curandState *state);

__device__ CudaColor ray_color_cuda(const CudaRay &ray, int depth,
                                    const CudaHittable *world,
                                    const CudaHittable *lights,
                                    CudaColor background, curandState *state);

#endif // USE_CUDA