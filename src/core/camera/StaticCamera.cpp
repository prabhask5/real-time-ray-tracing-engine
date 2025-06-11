#include "StaticCamera.hpp"
#include "../../optimization/BVHNode.hpp"
#include "../HittableList.hpp"
#include "../Ray.hpp"
#include <ColorUtility.hpp>
#include <ThreadPool.hpp>
#include <Vec3Utility.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

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
  // TODO: Implement CUDA-based rendering path.
  // This skeleton is provided for future GPU acceleration.
  // For now, fall back to CPU rendering.
  render_cpu(world, lights);
}