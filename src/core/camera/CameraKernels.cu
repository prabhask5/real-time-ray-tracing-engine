#ifdef USE_CUDA

#include "../../scene/materials/Material.cuh"
#include "../../utils/math/PDF.cuh"
#include "../../utils/math/Utility.cuh"
#include "../../utils/math/Vec3Utility.cuh"
#include "../ScatterRecord.cuh"
#include "CameraKernels.cuh"

// Initialize random states for each pixel.

__global__ void init_rand_states(curandState *rand_states, int width,
                                 int height, unsigned long seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= width || j >= height)
    return;

  int pixel_index = j * width + i;
  curand_init(seed + pixel_index, pixel_index, 0, &rand_states[pixel_index]);
}

// Device function: Sample stratified square.

__device__ CudaVec3 sample_square_stratified_cuda(int s_i, int s_j,
                                                  int sqrt_spp,
                                                  curandState *state) {
  double px = ((double)s_i + cuda_random_double(state)) / sqrt_spp - 0.5;
  double py = ((double)s_j + cuda_random_double(state)) / sqrt_spp - 0.5;
  return cuda_make_vec3(px, py, 0.0);
}

// Device function: Sample random square.

__device__ CudaVec3 sample_square_cuda(curandState *state) {
  return cuda_make_vec3(cuda_random_double(state) - 0.5,
                        cuda_random_double(state) - 0.5, 0.0);
}

// Device function: Sample disk.

__device__ CudaVec3 sample_disk_cuda(double radius, curandState *state) {
  return radius * cuda_vec3_random_in_unit_disk(state);
}

// Device function: Sample defocus disk.

__device__ CudaPoint3 defocus_disk_sample_cuda(CudaVec3 defocus_disk_u,
                                               CudaVec3 defocus_disk_v,
                                               curandState *state) {
  CudaVec3 p = cuda_vec3_random_in_unit_disk(state);
  return p.x * defocus_disk_u + p.y * defocus_disk_v;
}

// Device function: Get ray.

__device__ CudaRay get_ray_cuda(int i, int j, int s_i, int s_j, int sqrt_spp,
                                CudaPoint3 center, CudaPoint3 pixel00_loc,
                                CudaVec3 pixel_delta_u, CudaVec3 pixel_delta_v,
                                CudaVec3 defocus_disk_u,
                                CudaVec3 defocus_disk_v, double defocus_angle,
                                curandState *state) {
  // Computes a jittered offset within the pixel grid cell defined by (s_i,
  // s_j). This random offset ensures that each ray passes through a different
  // point inside its subpixel, enabling stratified sampling and reducing
  // aliasing artifacts. Without this, all rays would pass through the exact
  // center of their subpixels.
  CudaVec3 offset = sample_square_stratified_cuda(s_i, s_j, sqrt_spp, state);

  // This calculates the 3D world-space point that corresponds to pixel (i, j),
  // with subpixel jitter.
  CudaPoint3 pixel_sample = pixel00_loc + (i + offset.x) * pixel_delta_u +
                            (j + offset.y) * pixel_delta_v;

  // If defocus_angle == 0: use the camera center -> behaves like a pinhole
  // camera. Else: use a point sampled from a disk, simulating a lens with
  // radius and blur (depth of field).
  CudaPoint3 ray_origin =
      (defocus_angle <= 0)
          ? center
          : defocus_disk_sample_cuda(defocus_disk_u, defocus_disk_v, state);

  // Determine the ray direction by finding the vector difference between the
  // pixel sample and the ray origin.
  CudaVec3 ray_direction = pixel_sample - ray_origin;

  double ray_time = cuda_random_double(state);

  // Returns the resulting ray from the origin and direction.
  return cuda_make_ray(ray_origin, ray_direction, ray_time);
}

// Forward declaration.

__device__ CudaColor ray_color_cuda(const CudaRay &ray, int depth,
                                    const CudaHittable &world,
                                    const CudaHittable &lights,
                                    CudaColor background, curandState *state);

// Device function: Ray color computation with proper material handling.

__device__ CudaColor ray_color_cuda(const CudaRay &ray, int depth,
                                    const CudaHittable &world,
                                    const CudaHittable &lights,
                                    CudaColor background, curandState *state) {
  // Base case: if we've exceeded the ray bounce limit, no more light is
  // gathered.
  if (depth <= 0)
    return cuda_make_vec3(0.0, 0.0, 0.0);

  CudaHitRecord rec;

  // If the ray hits nothing, return the background color.
  if (!cuda_hittable_hit(world, ray, cuda_make_interval(0.001, CUDA_INF), rec,
                         state))
    return background;

  // If the ray hits a hittable objct, we store the hit record and calculate the
  // scattering based on the material.

  // Get material from hit record.
  const CudaMaterial *material = rec.material;

  // Calculate emission color.
  CudaColor color_from_emission =
      cuda_material_emitted(*material, ray, rec, rec.u, rec.v, rec.point);

  // Try to scatter the ray.
  CudaScatterRecord srec;

  // If the material does not scatter, we just return the material's emitted
  // color for that point.
  if (!cuda_material_scatter(*material, ray, rec, srec, state))
    return color_from_emission;

  // If the material scatters the ray in a specific direction without using a
  // PDF, follow that ray recursively and scale the result by the attenuation.
  // This ONLY happens when the material chooses to bypass the PDF sampling
  // logic.
  if (srec.skip_pdf) {
    return srec.attenuation * ray_color_cuda(srec.skip_pdf_ray, depth - 1,
                                             world, lights, background, state);
  }

  // Create a PDF that samples directions toward the light sources from the hit
  // point. This helps concentrate rays toward bright areas for better rendering
  // efficiency.
  // NOTE: In the case that the lights hittable list object is empty (there's no
  // lights), we cannot use scattering from light sources, so we skip.
  //
  // Combine two PDFs: one from the material's scattering function and one
  // toward the lights. This is a mixture of importance sampling strategies to
  // improve convergence.
  CudaPDF mixture_pdf;
  if (lights.type != CudaHittableType::HITTABLE_LIST ||
      lights.hittable_list->count == 0)
    mixture_pdf = cuda_make_pdf_mixture(srec.pdf, srec.pdf);
  else {
    CudaPDF hittable_pdf = cuda_make_pdf_hittable(&lights, rec.point);
    mixture_pdf = cuda_make_pdf_mixture(&hittable_pdf, srec.pdf);
  }

  // Sample from the mixture PDF.
  CudaVec3 scattered_direction = cuda_pdf_generate(mixture_pdf, state);

  // Compute the probability density function (PDF) value of the sampled
  // direction. This is used to correctly scale the Monte Carlo estimate.
  double pdf_value = cuda_pdf_value(mixture_pdf, scattered_direction);

  // Generate a new scattered ray based on the mixture PDF.
  // This ray originates from the hit point and is aimed in a sampled direction.
  CudaRay scattered_ray =
      cuda_make_ray(rec.point, scattered_direction, ray.time);

  // Ask the material what its theoretical scattering PDF is for this
  // interaction. This is needed for physically-based importance sampling.
  double scattering_pdf =
      cuda_material_scattering_pdf(*material, ray, rec, scattered_ray);

  // Scale the returned color by:
  /// - `attenuation`: how much the material reduces the ray's energy.
  /// - `scattering_pdf`: how likely the material is to scatter in that
  /// direction.
  /// - `1 / pdf_value`: normalizes the estimate based on how likely we were to
  /// choose this ray.
  CudaColor color_from_scatter =
      (srec.attenuation * scattering_pdf *
       ray_color_cuda(scattered_ray, depth - 1, world, lights, background,
                      state)) /
      pdf_value;

  // Add any emitted light from the surface (like glowing surfaces) to the
  // scattered result.
  return color_from_emission + color_from_scatter;
}

// Dynamic camera tile rendering kernel.

__global__ void dynamic_render_tile_kernel(
    CudaColor *accumulation, int image_width, int image_height, int start_r,
    int end_r, int start_c, int end_c, int s_i, int s_j, int sqrt_spp,
    int max_depth, CudaPoint3 center, CudaPoint3 pixel00_loc,
    CudaVec3 pixel_delta_u, CudaVec3 pixel_delta_v, CudaVec3 u, CudaVec3 v,
    CudaVec3 w, CudaVec3 defocus_disk_u, CudaVec3 defocus_disk_v,
    double defocus_angle, CudaColor background, CudaHittable world,
    CudaHittable lights, curandState *rand_states) {

  int i = blockIdx.x * blockDim.x + threadIdx.x + start_c;
  int j = blockIdx.y * blockDim.y + threadIdx.y + start_r;

  if (i >= end_c || j >= end_r)
    return;

  int pixel_index = j * image_width + i;
  curandState *state = &rand_states[pixel_index];

  CudaRay ray = get_ray_cuda(i, j, s_i, s_j, sqrt_spp, center, pixel00_loc,
                             pixel_delta_u, pixel_delta_v, defocus_disk_u,
                             defocus_disk_v, defocus_angle, state);

  CudaColor sample =
      ray_color_cuda(ray, max_depth, world, lights, background, state);

  // Atomic add to accumulation buffer.

  atomicAdd(&accumulation[pixel_index].x, sample.x);
  atomicAdd(&accumulation[pixel_index].y, sample.y);
  atomicAdd(&accumulation[pixel_index].z, sample.z);
}

// Static camera rendering kernel.

__global__ void static_render_kernel(
    CudaColor *pixel_colors, int image_width, int image_height, int start_row,
    int end_row, int sqrt_spp, int max_depth, CudaPoint3 center,
    CudaPoint3 pixel00_loc, CudaVec3 pixel_delta_u, CudaVec3 pixel_delta_v,
    CudaVec3 u, CudaVec3 v, CudaVec3 w, CudaVec3 defocus_disk_u,
    CudaVec3 defocus_disk_v, double defocus_angle, double pixel_samples_scale,
    CudaColor background, CudaHittable world, CudaHittable lights,
    curandState *rand_states) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y + start_row;

  if (i >= image_width || j >= end_row)
    return;

  int pixel_index = j * image_width + i;
  curandState *state = &rand_states[pixel_index];

  CudaColor pixel_color = cuda_make_vec3(0, 0, 0);

  // Sample all stratified samples for this pixel.

  for (int s_j_idx = 0; s_j_idx < sqrt_spp; ++s_j_idx) {
    for (int s_i_idx = 0; s_i_idx < sqrt_spp; ++s_i_idx) {
      CudaRay ray = get_ray_cuda(
          i, j, s_i_idx, s_j_idx, sqrt_spp, center, pixel00_loc, pixel_delta_u,
          pixel_delta_v, defocus_disk_u, defocus_disk_v, defocus_angle, state);

      pixel_color =
          cuda_vec3_add(pixel_color, ray_color_cuda(ray, max_depth, world,
                                                    lights, background, state));
    }
  }

  // Scale and store final pixel color.

  pixel_colors[pixel_index] = pixel_samples_scale * pixel_color;
}

#endif // USE_CUDA