#ifdef USE_CUDA

#include "CameraKernelWrappers.cuh"

void cuda_init_rand_states_wrapper(curandState *d_rand_states, int width,
                                   int height, unsigned long seed,
                                   dim3 grid_size, dim3 block_size) {
  init_rand_states<<<grid_size, block_size>>>(d_rand_states, width, height,
                                              seed);
}

void cuda_dynamic_render_tile_wrapper(
    CudaColor *accumulation, int image_width, int image_height, int start_r,
    int end_r, int start_c, int end_c, int s_i, int s_j, int sqrt_spp,
    int max_depth, CudaPoint3 center, CudaPoint3 pixel00_loc,
    CudaVec3 pixel_delta_u, CudaVec3 pixel_delta_v, CudaVec3 u, CudaVec3 v,
    CudaVec3 w, CudaVec3 defocus_disk_u, CudaVec3 defocus_disk_v,
    double defocus_angle, CudaColor background, CudaHittable world,
    CudaHittable lights, curandState *rand_states, dim3 grid_size,
    dim3 block_size) {

  dynamic_render_tile_kernel<<<grid_size, block_size>>>(
      accumulation, image_width, image_height, start_r, end_r, start_c, end_c,
      s_i, s_j, sqrt_spp, max_depth, center, pixel00_loc, pixel_delta_u,
      pixel_delta_v, u, v, w, defocus_disk_u, defocus_disk_v, defocus_angle,
      background, world, lights, rand_states);
}

void cuda_static_render_wrapper(
    CudaColor *pixel_colors, int image_width, int image_height, int start_row,
    int end_row, int sqrt_spp, int max_depth, CudaPoint3 center,
    CudaPoint3 pixel00_loc, CudaVec3 pixel_delta_u, CudaVec3 pixel_delta_v,
    CudaVec3 u, CudaVec3 v, CudaVec3 w, CudaVec3 defocus_disk_u,
    CudaVec3 defocus_disk_v, double defocus_angle, double pixel_samples_scale,
    CudaColor background, CudaHittable world, CudaHittable lights,
    curandState *rand_states, dim3 grid_size, dim3 block_size) {

  static_render_kernel<<<grid_size, block_size>>>(
      pixel_colors, image_width, image_height, start_row, end_row, sqrt_spp,
      max_depth, center, pixel00_loc, pixel_delta_u, pixel_delta_v, u, v, w,
      defocus_disk_u, defocus_disk_v, defocus_angle, pixel_samples_scale,
      background, world, lights, rand_states);
}

#endif // USE_CUDA