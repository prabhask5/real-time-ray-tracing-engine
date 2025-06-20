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
#include "../../scene/SceneConversions.cuh"
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
  dim3 block_size(16, 16);
  dim3 grid_size((m_image_width + block_size.x - 1) / block_size.x,
                 (m_image_height + block_size.y - 1) / block_size.y);

  init_rand_states<<<grid_size, block_size>>>(d_rand_states, m_image_width,
                                              m_image_height,
                                              (unsigned long)time(nullptr));
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "Failed to initialize CUDA random states" << std::endl;
    cudaFree(d_pixel_colors);
    cudaFree(d_rand_states);
    render_cpu(world, lights);
    return;
  }

  // Convert CPU objects to CUDA format using comprehensive conversion.
  CudaSceneData cuda_scene_data = convert_complete_scene_to_cuda(world, lights);
  if (cuda_scene_data.world_objects_buffer == nullptr ||
      cuda_scene_data.lights_objects_buffer == nullptr) {
    std::cerr << "Failed to convert scene to CUDA format" << std::endl;
    cudaFree(d_pixel_colors);
    cudaFree(d_rand_states);
    render_cpu(world, lights);
    return;
  }
  CudaHittableList cuda_world = cuda_scene_data.world;
  CudaHittableList cuda_lights = cuda_scene_data.lights;

  int sqrt_spp = static_cast<int>(std::sqrt(m_samples_per_pixel));

  // Convert camera parameters to CUDA format.
  CudaPoint3 cuda_center(m_center.x(), m_center.y(), m_center.z());
  CudaPoint3 cuda_pixel00_loc(m_pixel00_loc.x(), m_pixel00_loc.y(),
                              m_pixel00_loc.z());
  CudaVec3 cuda_pixel_delta_u(m_pixel_delta_u.x(), m_pixel_delta_u.y(),
                              m_pixel_delta_u.z());
  CudaVec3 cuda_pixel_delta_v(m_pixel_delta_v.x(), m_pixel_delta_v.y(),
                              m_pixel_delta_v.z());
  CudaVec3 cuda_u(m_u.x(), m_u.y(), m_u.z());
  CudaVec3 cuda_v(m_v.x(), m_v.y(), m_v.z());
  CudaVec3 cuda_w(m_w.x(), m_w.y(), m_w.z());
  CudaVec3 cuda_defocus_disk_u(m_defocus_disk_u.x(), m_defocus_disk_u.y(),
                               m_defocus_disk_u.z());
  CudaVec3 cuda_defocus_disk_v(m_defocus_disk_v.x(), m_defocus_disk_v.y(),
                               m_defocus_disk_v.z());
  CudaColor cuda_background(m_background.x(), m_background.y(),
                            m_background.z());

  // Process rows in batches to avoid memory issues.
  const int batch_size = 64; // Process 64 rows at a time
  for (int start_row = 0; start_row < m_image_height; start_row += batch_size) {
    std::clog << "\rScanlines remaining: " << (m_image_height - start_row)
              << ' ' << std::flush;

    int end_row = std::min(start_row + batch_size, m_image_height);

    // Clear pixel colors for this batch.
    cudaMemset(d_pixel_colors + start_row * m_image_width, 0,
               (end_row - start_row) * m_image_width * sizeof(CudaColor));

    // Launch CUDA kernel for this batch.
    dim3 batch_grid_size((m_image_width + block_size.x - 1) / block_size.x,
                         (end_row - start_row + block_size.y - 1) /
                             block_size.y);

    static_render_kernel<<<batch_grid_size, block_size>>>(
        d_pixel_colors, m_image_width, m_image_height, start_row, end_row,
        sqrt_spp, m_max_depth, cuda_center, cuda_pixel00_loc,
        cuda_pixel_delta_u, cuda_pixel_delta_v, cuda_u, cuda_v, cuda_w,
        cuda_defocus_disk_u, cuda_defocus_disk_v, m_defocus_angle,
        m_pixel_samples_scale, cuda_background, cuda_world, cuda_lights,
        d_rand_states);

    cudaDeviceSynchronize();

    // Copy results back to CPU and write to file.
    std::vector<CudaColor> batch_colors((end_row - start_row) * m_image_width);
    cudaMemcpy(batch_colors.data(), d_pixel_colors + start_row * m_image_width,
               (end_row - start_row) * m_image_width * sizeof(CudaColor),
               cudaMemcpyDeviceToHost);

    // Write batch to file.
    for (int row = 0; row < (end_row - start_row); ++row) {
      for (int col = 0; col < m_image_width; ++col) {
        int idx = row * m_image_width + col;
        Color cpu_color(batch_colors[idx].x, batch_colors[idx].y,
                        batch_colors[idx].z);
        write_color(image, cpu_color);
      }
    }
  }

  std::clog << "\rDone.                 \n";

  // Cleanup CUDA memory and scene data.
  cleanup_cuda_scene_data(cuda_scene_data);
  cudaFree(d_pixel_colors);
  cudaFree(d_rand_states);
#else
  // Fall back to CPU rendering if CUDA is not available.
  render_cpu(world, lights);
#endif
}