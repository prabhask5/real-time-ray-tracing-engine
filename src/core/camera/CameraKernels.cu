#ifdef USE_CUDA

#include "../../utils/math/Utility.cuh"
#include "../../utils/math/Vec3Utility.cuh"
#include "CameraKernels.cuh"

// Initialize random states for each pixel
__global__ void init_rand_states(curandState *rand_states, int width,
                                 int height, unsigned long seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= width || j >= height)
    return;

  int pixel_index = j * width + i;
  curand_init(seed + pixel_index, pixel_index, 0, &rand_states[pixel_index]);
}

// Device function: Sample stratified square
__device__ CudaVec3 sample_square_stratified_cuda(int s_i, int s_j,
                                                  int sqrt_spp,
                                                  curandState *state) {
  double px = ((double)s_i + curand_uniform_double(state)) / sqrt_spp - 0.5;
  double py = ((double)s_j + curand_uniform_double(state)) / sqrt_spp - 0.5;
  return CudaVec3(px, py, 0);
}

// Device function: Sample random square
__device__ CudaVec3 sample_square_cuda(curandState *state) {
  return CudaVec3(curand_uniform_double(state) - 0.5,
                  curand_uniform_double(state) - 0.5, 0);
}

// Device function: Sample disk
__device__ CudaVec3 sample_disk_cuda(double radius, curandState *state) {
  CudaVec3 p;
  do {
    p = 2.0 * CudaVec3(curand_uniform_double(state),
                       curand_uniform_double(state), 0) -
        CudaVec3(1, 1, 0);
  } while (dot(p, p) >= 1.0);
  return radius * p;
}

// Device function: Sample defocus disk
__device__ CudaPoint3 defocus_disk_sample_cuda(CudaVec3 defocus_disk_u,
                                               CudaVec3 defocus_disk_v,
                                               curandState *state) {
  CudaVec3 p = sample_disk_cuda(1.0, state);
  return p.x() * defocus_disk_u + p.y() * defocus_disk_v;
}

// Device function: Get ray
__device__ CudaRay get_ray_cuda(int i, int j, int s_i, int s_j, int sqrt_spp,
                                CudaPoint3 center, CudaPoint3 pixel00_loc,
                                CudaVec3 pixel_delta_u, CudaVec3 pixel_delta_v,
                                CudaVec3 defocus_disk_u,
                                CudaVec3 defocus_disk_v, double defocus_angle,
                                curandState *state) {
  CudaVec3 offset = sample_square_stratified_cuda(s_i, s_j, sqrt_spp, state);
  CudaPoint3 pixel_sample = pixel00_loc + (i + offset.x()) * pixel_delta_u +
                            (j + offset.y()) * pixel_delta_v;

  CudaPoint3 ray_origin =
      (defocus_angle <= 0)
          ? center
          : center +
                defocus_disk_sample_cuda(defocus_disk_u, defocus_disk_v, state);
  CudaVec3 ray_direction = pixel_sample - ray_origin;
  double ray_time = curand_uniform_double(state);

  return CudaRay(ray_origin, ray_direction, ray_time);
}

// Forward declaration
__device__ CudaColor ray_color_cuda(const CudaRay &ray, int depth,
                                    const CudaHittableList &world,
                                    const CudaHittableList &lights,
                                    CudaColor background, curandState *state);

// Device function: Ray color computation with proper material handling
__device__ CudaColor ray_color_cuda(const CudaRay &ray, int depth,
                                    const CudaHittableList &world,
                                    const CudaHittableList &lights,
                                    CudaColor background, curandState *state) {
  if (depth <= 0)
    return CudaColor(0, 0, 0);

  CudaHitRecord rec;
  if (!world.hit(ray, CudaInterval(0.001, CUDA_INF), rec, state))
    return background;

  // Extract material from hit record
  if (rec.material_data == nullptr) {
    // Default diffuse behavior if no material
    CudaVec3 scatter_direction = rec.normal + cuda_random_unit_vector(state);
    if (scatter_direction.near_zero())
      scatter_direction = rec.normal;

    CudaRay scattered(rec.point, scatter_direction, ray.time);
    CudaColor attenuation = CudaColor(0.7, 0.7, 0.7);
    return attenuation * ray_color_cuda(scattered, depth - 1, world, lights,
                                        background, state);
  }

  // Get material from hit record
  const CudaMaterial *material =
      reinterpret_cast<const CudaMaterial *>(rec.material_data);

  // Calculate emission color
  CudaColor color_from_emission =
      material->emitted(ray, rec, rec.u, rec.v, rec.point);

  // Try to scatter the ray
  CudaScatterRecord srec;
  if (!material->scatter(ray, rec, srec, state))
    return color_from_emission;

  // Handle deterministic scattering (like perfect reflections)
  if (srec.skip_pdf) {
    return srec.attenuation * ray_color_cuda(srec.skip_pdf_ray, depth - 1,
                                             world, lights, background, state);
  }

  CudaColor color_from_scatter = CudaColor(0, 0, 0);

  // Handle probabilistic scattering with PDF
  if (lights.count > 0) {
    // Use importance sampling with lights
    CudaPDF light_pdf = cuda_make_hittable_pdf(&lights, rec.point);

    // Create complete mixed PDF with proper importance sampling
    CudaVec3 scattered_direction;
    double pdf_value;

    // Create material PDF based on scatter record type
    CudaPDF material_pdf;
    if (srec.pdf_type == CUDA_PDF_COSINE) {
      material_pdf = cuda_make_cosine_pdf(rec.normal);
    } else if (srec.pdf_type == CUDA_PDF_SPHERE) {
      material_pdf = cuda_make_sphere_pdf();
    } else {
      material_pdf = cuda_make_cosine_pdf(rec.normal); // Default fallback
    }

    // Create proper mixture PDF combining light and material PDFs
    CudaPDF mixture_pdf =
        cuda_make_mixture_pdf(light_pdf.type, (void *)&light_pdf.data,
                              material_pdf.type, (void *)&material_pdf.data);

    // Sample from the mixture PDF
    scattered_direction = mixture_pdf.generate(state);
    pdf_value = mixture_pdf.value(scattered_direction);

    // Ensure PDF value is valid
    if (pdf_value <= 0.0) {
      // Fallback to material PDF only
      scattered_direction = material_pdf.generate(state);
      pdf_value = material_pdf.value(scattered_direction);
    }

    if (pdf_value > 0.0) {
      CudaRay scattered_ray(rec.point, scattered_direction, ray.time);
      double scattering_pdf = material->scattering_pdf(ray, rec, scattered_ray);
      color_from_scatter = (srec.attenuation * scattering_pdf *
                            ray_color_cuda(scattered_ray, depth - 1, world,
                                           lights, background, state)) /
                           pdf_value;
    }
  } else {
    // No lights - use material PDF only
    if (srec.pdf_type == CUDA_PDF_COSINE) {
      CudaPDF material_pdf = cuda_make_cosine_pdf(rec.normal);
      CudaVec3 scattered_direction = material_pdf.generate(state);
      double pdf_value = material_pdf.value(scattered_direction);

      if (pdf_value > 0.0) {
        CudaRay scattered_ray(rec.point, scattered_direction, ray.time);
        double scattering_pdf =
            material->scattering_pdf(ray, rec, scattered_ray);
        color_from_scatter = (srec.attenuation * scattering_pdf *
                              ray_color_cuda(scattered_ray, depth - 1, world,
                                             lights, background, state)) /
                             pdf_value;
      }
    } else {
      // Fallback to hemisphere sampling
      CudaVec3 scattered_direction =
          cuda_random_on_hemisphere(state, rec.normal);
      CudaRay scattered_ray(rec.point, scattered_direction, ray.time);
      color_from_scatter =
          srec.attenuation * ray_color_cuda(scattered_ray, depth - 1, world,
                                            lights, background, state);
    }
  }

  return color_from_emission + color_from_scatter;
}

// Dynamic camera tile rendering kernel
__global__ void dynamic_render_tile_kernel(
    CudaColor *accumulation, int image_width, int image_height, int start_r,
    int end_r, int start_c, int end_c, int s_i, int s_j, int sqrt_spp,
    int max_depth, CudaPoint3 center, CudaPoint3 pixel00_loc,
    CudaVec3 pixel_delta_u, CudaVec3 pixel_delta_v, CudaVec3 u, CudaVec3 v,
    CudaVec3 w, CudaVec3 defocus_disk_u, CudaVec3 defocus_disk_v,
    double defocus_angle, CudaColor background, CudaHittableList world,
    CudaHittableList lights, curandState *rand_states) {

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

  // Atomic add to accumulation buffer
  atomicAdd(&accumulation[pixel_index].x, sample.x);
  atomicAdd(&accumulation[pixel_index].y, sample.y);
  atomicAdd(&accumulation[pixel_index].z, sample.z);
}

// Static camera rendering kernel
__global__ void static_render_kernel(
    CudaColor *pixel_colors, int image_width, int image_height, int start_row,
    int end_row, int sqrt_spp, int max_depth, CudaPoint3 center,
    CudaPoint3 pixel00_loc, CudaVec3 pixel_delta_u, CudaVec3 pixel_delta_v,
    CudaVec3 u, CudaVec3 v, CudaVec3 w, CudaVec3 defocus_disk_u,
    CudaVec3 defocus_disk_v, double defocus_angle, double pixel_samples_scale,
    CudaColor background, CudaHittableList world, CudaHittableList lights,
    curandState *rand_states) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y + start_row;

  if (i >= image_width || j >= end_row)
    return;

  int pixel_index = j * image_width + i;
  curandState *state = &rand_states[pixel_index];

  CudaColor pixel_color(0, 0, 0);

  // Sample all stratified samples for this pixel
  for (int s_j_idx = 0; s_j_idx < sqrt_spp; ++s_j_idx) {
    for (int s_i_idx = 0; s_i_idx < sqrt_spp; ++s_i_idx) {
      CudaRay ray = get_ray_cuda(
          i, j, s_i_idx, s_j_idx, sqrt_spp, center, pixel00_loc, pixel_delta_u,
          pixel_delta_v, defocus_disk_u, defocus_disk_v, defocus_angle, state);

      pixel_color +=
          ray_color_cuda(ray, max_depth, world, lights, background, state);
    }
  }

  // Scale and store final pixel color
  pixel_colors[pixel_index] = pixel_samples_scale * pixel_color;
}

#endif // USE_CUDA