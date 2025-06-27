#include "StaticCamera.hpp"
#include "../../optimization/BVHNode.hpp"
#include "../../utils/ColorUtility.hpp"
#include "../../utils/concurrency/ThreadPool.hpp"
#include "../../utils/math/Vec3Utility.hpp"
#include "../HittableList.hpp"
#include "../Ray.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef USE_CUDA
#include "../../scene/CudaSceneInitialization.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "../../utils/memory/CudaMemoryUtility.cuh"
#include "CameraKernelWrappers.cuh"
#include "CameraKernels.cuh"
#include <ctime>
#endif

StaticCamera::StaticCamera(const CameraConfig &config,
                           std::string output_file_name)
    : Camera(config), m_output_file(output_file_name) {}

void StaticCamera::render(HittableList &world, HittableList &lights) {
  if (m_use_gpu)
    render_gpu(world, lights);
  else
    render_cpu(world, lights);
}

void StaticCamera::render_cpu(HittableList &world, HittableList &lights) {
  initialize();

  if (m_use_bvh) {
    if (!world.get_objects().empty())
      world = HittableList(std::make_shared<BVHNode>(world));
    if (!lights.get_objects().empty())
      lights = HittableList(std::make_shared<BVHNode>(lights));
  }

  if (!std::filesystem::exists("output"))
    std::filesystem::create_directories("output");

  std::ofstream image("output/" + m_output_file,
                      std::ios::out | std::ios::trunc);
  if (!image) {
    std::cerr << "Failed to open output/image.ppm for writing." << std::endl;
    return;
  }
  image << "P3\n" << m_image_width << ' ' << m_image_height << "\n255\n";

  if (m_use_parallelism) {
    ThreadPool pool(std::thread::hardware_concurrency());
    std::vector<Color> row_colors(m_image_width);

    for (int j = 0; j < m_image_height; ++j) {
      std::clog << "\rScanlines remaining: " << (m_image_height - j) << ' '
                << std::flush;
      pool.start();

      for (int i = 0; i < m_image_width; ++i) {
        pool.submit_job([this, &world, &lights, i, j, &row_colors]() {
          row_colors[i] = Color(0, 0, 0);

          // Shoots m_samples_per_pixel rays per pixel in a stratified grid
          // pattern.
          int sqrt_spp = static_cast<int>(std::sqrt(m_samples_per_pixel));
          for (int s_j = 0; s_j < sqrt_spp; ++s_j) {
            for (int s_i = 0; s_i < sqrt_spp; ++s_i) {
              // Shoots a ray through a random subpixel region, stratified by
              // s_i
              // and s_j. This simulates camera defocus (depth of field) and
              // performs anti-aliasing by jittering the ray within the pixel
              // grid.
              Ray ray = get_ray(i, j, s_i, s_j);

              // Traces the ray and accumulates the color contribution from this
              // sample.
              row_colors[i] += ray_color(ray, m_max_depth, world, lights);
            }
          }
        });
      }

      pool.finish();

      for (const Color &c : row_colors) {
        // Normalizes the summed pixel colors (averages by the number of
        // samples) and converts to 8-bit [0,255] RGB and writes to the .ppm
        // file.
        write_color(image, m_pixel_samples_scale * c);
      }
    }
  } else {
    for (int j = 0; j < m_image_height; ++j) {
      std::clog << "\rScanlines remaining: " << (m_image_height - j) << ' '
                << std::flush;
      for (int i = 0; i < m_image_width; ++i) {
        Color pixel_color(0, 0, 0);

        // Shoots m_samples_per_pixel rays per pixel in a stratified grid
        // pattern.
        int sqrt_spp = static_cast<int>(std::sqrt(m_samples_per_pixel));
        for (int s_j = 0; s_j < sqrt_spp; ++s_j) {
          for (int s_i = 0; s_i < sqrt_spp; ++s_i) {
            // Shoots a ray through a random subpixel region, stratified by s_i
            // and s_j. This simulates camera defocus (depth of field) and
            // performs anti-aliasing by jittering the ray within the pixel
            // grid.
            Ray ray = get_ray(i, j, s_i, s_j);

            // Traces the ray and accumulates the color contribution from this
            // sample.
            pixel_color += ray_color(ray, m_max_depth, world, lights);
          }
        }

        // Normalizes the summed pixel colors (averages by the number of
        // samples)
        // and converts to 8-bit [0,255] RGB and writes to the .ppm file.
        write_color(image, m_pixel_samples_scale * pixel_color);
      }
    }
  }

  std::clog << "\rDone.                 \n";
}

void StaticCamera::render_gpu(HittableList &world, HittableList &lights) {
#ifdef USE_CUDA
  initialize();

  if (m_use_bvh) {
    if (!world.get_objects().empty())
      world = HittableList(std::make_shared<BVHNode>(world));
    if (!lights.get_objects().empty())
      lights = HittableList(std::make_shared<BVHNode>(lights));
  }

  if (!std::filesystem::exists("output"))
    std::filesystem::create_directories("output");

  std::ofstream image("output/" + m_output_file,
                      std::ios::out | std::ios::trunc);
  if (!image) {
    std::cerr << "Failed to open output/" << m_output_file << " for writing."
              << std::endl;
    return;
  }
  image << "P3\n" << m_image_width << ' ' << m_image_height << "\n255\n";

  // CUDA setup.
  CudaColor *d_pixel_colors;
  curandState *d_rand_states;
  size_t colors_size = m_image_width * m_image_height * sizeof(CudaColor);
  size_t rand_states_size =
      m_image_width * m_image_height * sizeof(curandState);

  if (cudaMalloc(&d_pixel_colors, colors_size) != cudaSuccess) {
    std::cerr << "Failed to allocate CUDA memory for pixel colors" << std::endl;
    render_cpu(world, lights);
    return;
  }
  if (cudaMalloc(&d_rand_states, rand_states_size) != cudaSuccess) {
    std::cerr << "Failed to allocate CUDA memory for random states"
              << std::endl;
    cudaFree(d_pixel_colors);
    render_cpu(world, lights);
    return;
  }

  // Initialize random states.
  dim3 block_size(16, 16); // 256 threads per block.
  dim3 grid_size((m_image_width + block_size.x - 1) / block_size.x,
                 (m_image_height + block_size.y - 1) /
                     block_size.y); // Blocks to fill image.

  cuda_init_rand_states_wrapper(d_rand_states, m_image_width, m_image_height,
                                (unsigned long)time(nullptr), grid_size,
                                block_size);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "Failed to initialize CUDA random states" << std::endl;
    cudaFree(d_pixel_colors);
    cudaFree(d_rand_states);
    render_cpu(world, lights);
    return;
  }

  // Check for CUDA errors before proceeding
  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess) {
    std::cerr << "CUDA error before scene init: "
              << cudaGetErrorString(cuda_error) << std::endl;
  }

  // Initialize CUDA scene using new comprehensive system.
  CudaSceneData cuda_scene_data = initialize_cuda_scene(world, lights);
  if (cuda_scene_data.world.get() == nullptr ||
      cuda_scene_data.lights.get() == nullptr) {
    std::cerr << "Failed to initialize CUDA scene" << std::endl;
    cudaFree(d_pixel_colors);
    cudaFree(d_rand_states);
    render_cpu(world, lights);
    return;
  }

  // Copy CudaHittable structs from device to host for kernel launch.
  CudaHittable cuda_world, cuda_lights;
  cudaMemcpyDeviceToHostSafe(&cuda_world, cuda_scene_data.world.get(), 1);
  cudaMemcpyDeviceToHostSafe(&cuda_lights, cuda_scene_data.lights.get(), 1);

  int sqrt_spp = static_cast<int>(std::sqrt(m_samples_per_pixel));

  // Convert camera parameters to CUDA format.
  CudaPoint3 cuda_center = cpu_to_cuda_vec3(m_center);
  CudaPoint3 cuda_pixel00_loc = cpu_to_cuda_vec3(m_pixel00_loc);
  CudaVec3 cuda_pixel_delta_u = cpu_to_cuda_vec3(m_pixel_delta_u);
  CudaVec3 cuda_pixel_delta_v = cpu_to_cuda_vec3(m_pixel_delta_v);
  CudaVec3 cuda_u = cpu_to_cuda_vec3(m_u);
  CudaVec3 cuda_v = cpu_to_cuda_vec3(m_v);
  CudaVec3 cuda_w = cpu_to_cuda_vec3(m_w);
  CudaVec3 cuda_defocus_disk_u = cpu_to_cuda_vec3(m_defocus_disk_u);
  CudaVec3 cuda_defocus_disk_v = cpu_to_cuda_vec3(m_defocus_disk_v);
  CudaColor cuda_background = cpu_to_cuda_vec3(m_background);

  // Process rows in batches to avoid memory issues.
  const int batch_size = 64; // Process 64 rows at a time.
  for (int start_row = 0; start_row < m_image_height; start_row += batch_size) {
    std::clog << "\rScanlines remaining: " << (m_image_height - start_row)
              << ' ' << std::flush;

    int end_row = std::min(start_row + batch_size, m_image_height);

    // Clear pixel colors for this batch.
    if (cudaMemset(d_pixel_colors + start_row * m_image_width, 0,
                   (end_row - start_row) * m_image_width * sizeof(CudaColor)) !=
        cudaSuccess) {
      std::cerr << "Failed to clear GPU memory for batch " << start_row
                << std::endl;
      cleanup_cuda_scene(cuda_scene_data);
      cudaFree(d_pixel_colors);
      cudaFree(d_rand_states);
      render_cpu(world, lights);
      return;
    }

    // Launch CUDA kernel for this batch.
    dim3 batch_grid_size((m_image_width + block_size.x - 1) / block_size.x,
                         (end_row - start_row + block_size.y - 1) /
                             block_size.y);

    cuda_static_render_wrapper(
        d_pixel_colors, m_image_width, m_image_height, start_row, end_row,
        sqrt_spp, m_max_depth, cuda_center, cuda_pixel00_loc,
        cuda_pixel_delta_u, cuda_pixel_delta_v, cuda_u, cuda_v, cuda_w,
        cuda_defocus_disk_u, cuda_defocus_disk_v, m_defocus_angle,
        m_pixel_samples_scale, cuda_background, cuda_world, cuda_lights,
        d_rand_states, batch_grid_size, block_size);

    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "CUDA kernel execution failed at batch " << start_row
                << std::endl;
      cleanup_cuda_scene(cuda_scene_data);
      cudaFree(d_pixel_colors);
      cudaFree(d_rand_states);
      render_cpu(world, lights);
      return;
    }

    // Copy results back to CPU and write to file.
    std::vector<CudaColor> batch_colors((end_row - start_row) * m_image_width);
    if (cudaMemcpy(batch_colors.data(),
                   d_pixel_colors + start_row * m_image_width,
                   (end_row - start_row) * m_image_width * sizeof(CudaColor),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      std::cerr << "Failed to copy results from GPU at batch " << start_row
                << std::endl;
      cleanup_cuda_scene(cuda_scene_data);
      cudaFree(d_pixel_colors);
      cudaFree(d_rand_states);
      render_cpu(world, lights);
      return;
    }

    // Write batch to file.
    for (int row = 0; row < (end_row - start_row); ++row) {
      for (int col = 0; col < m_image_width; ++col) {
        int idx = row * m_image_width + col;
        write_color(image, cuda_to_cpu_vec3(batch_colors[idx]));
      }
    }
  }

  std::clog << "\rDone.                 \n";

  // Cleanup CUDA memory and scene data.
  cleanup_cuda_scene(cuda_scene_data);
  cudaFree(d_pixel_colors);
  cudaFree(d_rand_states);
#else
  // Fall back to CPU rendering if CUDA is not available.
  render_cpu(world, lights);
#endif
}