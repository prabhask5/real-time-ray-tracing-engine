#include "StaticCamera.hpp"
#include "Ray.hpp"
#include <ColorUtility.hpp>
#include <Vec3Utility.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

StaticCamera::StaticCamera(const CameraConfig &config,
                           std::string output_file_name)
    : Camera(config), m_output_file(output_file_name) {}

void StaticCamera::render(const Hittable &world, const Hittable &lights) {
  initialize();

  if (!std::filesystem::exists("output"))
    std::filesystem::create_directories("output");

  std::ofstream image("output/" + m_output_file,
                      std::ios::out | std::ios::trunc);
  if (!image) {
    std::cerr << "Failed to open output/image.ppm for writing." << std::endl;
    return;
  }
  image << "P3\n" << m_image_width << ' ' << m_image_height << "\n255\n";

  for (int j = 0; j < m_image_height; j++) {
    std::clog << "\rScanlines remaining: " << (m_image_height - j) << ' '
              << std::flush;
    for (int i = 0; i < m_image_width; i++) {
      Color pixel_color(0, 0, 0);

      // Shoots m_samples_per_pixel rays per pixel in a stratified grid pattern.
      int sqrt_spp = int(std::sqrt(m_samples_per_pixel));
      for (int s_j = 0; s_j < sqrt_spp; s_j++) {
        for (int s_i = 0; s_i < sqrt_spp; s_i++) {
          // Shoots a ray through a random subpixel region, stratified by s_i
          // and s_j. This simulates camera defocus (depth of field) and
          // performs anti-aliasing by jittering the ray within the pixel grid.
          Ray ray = get_ray(i, j, s_i, s_j);

          // Traces the ray and accumulates the color contribution from this
          // sample.
          pixel_color += ray_color(ray, m_max_depth, world, lights);
        }
      }

      // Normalizes the summed pixel colors (averages by the number of samples)
      // and converts to 8-bit [0,255] RGB and writes to the .ppm file.
      write_color(image, m_pixel_samples_scale * pixel_color);
    }
  }

  std::clog << "\rDone.                 \n";
}