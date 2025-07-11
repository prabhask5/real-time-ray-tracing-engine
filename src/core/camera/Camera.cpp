#include "Camera.hpp"
#include "../../scene/materials/Material.hpp"
#include "../../utils/ColorUtility.hpp"
#include "../../utils/math/PDF.hpp"
#include "../HitRecord.hpp"
#include "../HittableList.hpp"
#include "../ScatterRecord.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef USE_CUDA
#include "../../scene/materials/Material.cuh"
#include "../../scene/textures/Texture.cuh"
#include "../../utils/memory/CudaSceneContext.cuh"
#include "../Hittable.cuh"
#endif

Camera::Camera(const CameraConfig &config)
    : m_aspect_ratio(config.aspect_ratio), m_image_width(config.image_width),
      m_samples_per_pixel(config.samples_per_pixel),
      m_max_depth(config.max_depth), m_background(config.background),
      m_vfov(config.vfov), m_lookfrom(config.lookfrom), m_lookat(config.lookat),
      m_vup(config.vup), m_defocus_angle(config.defocus_angle),
      m_focus_dist(config.focus_dist),
      m_use_parallelism(config.use_parallelism), m_use_bvh(config.use_bvh),
      m_use_gpu(config.use_gpu), m_use_debug(config.use_debug) {}

Camera::~Camera() {}

void Camera::initialize() {
  m_image_height = int(m_image_width / m_aspect_ratio);
  m_image_height = (m_image_height < 1) ? 1 : m_image_height;

  m_pixel_samples_scale = 1.0 / m_samples_per_pixel;

  m_center = m_lookfrom;

  // Determine viewport dimensions.
  double theta = degrees_to_radians(m_vfov);
  double h = std::tan(theta / 2);
  double viewport_height = 2 * h * m_focus_dist;
  double viewport_width =
      viewport_height * (double(m_image_width) / m_image_height);

  // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
  m_w = unit_vector(m_lookfrom - m_lookat);
  m_u = unit_vector(cross_product(m_vup, m_w));
  m_v = cross_product(m_w, m_u);

  // Calculate the vectors across the horizontal and down the vertical viewport
  // edges.
  Vec3 viewport_u =
      viewport_width * m_u; // Vector across viewport horizontal edge
  Vec3 viewport_v =
      viewport_height * -m_v; // Vector down viewport vertical edge

  // Calculate the horizontal and vertical delta vectors from pixel to pixel.
  m_pixel_delta_u = viewport_u / m_image_width;
  m_pixel_delta_v = viewport_v / m_image_height;

  // Calculate the location of the upper left pixel.
  Point3 viewport_upper_left =
      m_center - (m_focus_dist * m_w) - viewport_u / 2 - viewport_v / 2;
  m_pixel00_loc =
      viewport_upper_left + 0.5 * (m_pixel_delta_u + m_pixel_delta_v);

  // Calculate the camera defocus disk basis vectors.
  double defocus_radius =
      m_focus_dist * std::tan(degrees_to_radians(m_defocus_angle / 2));
  m_defocus_disk_u = m_u * defocus_radius;
  m_defocus_disk_v = m_v * defocus_radius;
}

void Camera::output_cpu_scene_json(const HittableList &obj,
                                   const std::string file_name) {
  if (!std::filesystem::exists("logs"))
    std::filesystem::create_directories("logs");

  std::clog << "[DEBUG] Printing JSON representation of CPU scene to "
            << "logs/" + file_name << std::endl;
  std::ofstream out("logs/" + file_name, std::ios::out | std::ios::trunc);
  out << obj.json();
  out.close();
}

#ifdef USE_CUDA
void Camera::output_cuda_scene_json(const CudaHittable &obj,
                                    const std::string file_name) {
  if (!std::filesystem::exists("logs"))
    std::filesystem::create_directories("logs");

  std::clog << "[DEBUG] Printing JSON representation of CUDA GPU scene to "
            << "logs/" + file_name << std::endl;
  std::ofstream out("logs/" + file_name, std::ios::out | std::ios::trunc);
  out << cuda_json_hittable(obj);
  out.close();
}

void Camera::output_cuda_scene_context_json() {
  // Get singleton scene context.
  CudaSceneContext &context = CudaSceneContext::get_context();

  if (!std::filesystem::exists("logs"))
    std::filesystem::create_directories("logs");

  std::clog << "[DEBUG] Printing JSON representation of CUDA Scene Context to "
               "logs/cuda_context_debug.json"
            << std::endl;

  std::ofstream out("logs/cuda_context_debug.json",
                    std::ios::out | std::ios::trunc);

  out << "{\n";
  out << "  \"type\": \"CudaSceneContext\",\n";
  out << "  \"material_count\": " << context.get_material_count() << ",\n";
  out << "  \"texture_count\": " << context.get_texture_count() << ",\n";

  // Output materials array with index mapping.
  out << "  \"materials\": [\n";
  const auto &materials = context.get_host_materials();
  for (size_t i = 0; i < materials.size(); ++i) {
    out << "    {\n";
    out << "      \"index\": " << i << ",\n";
    out << "      \"material\": " << cuda_json_material(materials[i]) << "\n";
    out << "    }";
    if (i < materials.size() - 1)
      out << ",";
    out << "\n";
  }
  out << "  ],\n";

  // Output textures array with index mapping.
  out << "  \"textures\": [\n";
  const auto &textures = context.get_host_textures();
  for (size_t i = 0; i < textures.size(); ++i) {
    out << "    {\n";
    out << "      \"index\": " << i << ",\n";
    out << "      \"texture\": " << cuda_json_texture(textures[i]) << "\n";
    out << "    }";
    if (i < textures.size() - 1)
      out << ",";
    out << "\n";
  }
  out << "  ]\n";

  out << "}\n";
  out.close();
}
#endif

Ray Camera::get_ray(int i, int j, int s_i, int s_j) const {
  // Computes a jittered offset within the pixel grid cell defined by (s_i,
  // s_j). This random offset ensures that each ray passes through a different
  // point inside its subpixel, enabling stratified sampling and reducing
  // aliasing artifacts. Without this, all rays would pass through the exact
  // center of their subpixels.
  Vec3 offset = sample_square_stratified(s_i, s_j);

#if SIMD_AVAILABLE && SIMD_DOUBLE_PRECISION
  // SIMD-optimized ray generation calculation.

  if constexpr (SIMD_DOUBLE_PRECISION) {
    // This calculates the 3D world-space point that corresponds to pixel (i,
    // j), with subpixel jitter. Uses SIMD-optimized Vec3 operations.
    Vec3 pixel_sample = m_pixel00_loc + ((i + offset.x()) * m_pixel_delta_u) +
                        ((j + offset.y()) * m_pixel_delta_v);

    // If defocus_angle == 0: use the camera center -> behaves like a pinhole
    // camera. Else: use a point sampled from a disk, simulating a lens with
    // radius and blur (depth of field).
    Point3 ray_origin =
        (m_defocus_angle <= 0) ? m_center : defocus_disk_sample();

    // Determine the ray direction by finding the vector difference between the
    // pixel sample and the ray origin. Uses SIMD-optimized Vec3 subtraction.
    Vec3 ray_direction = pixel_sample - ray_origin;

    double ray_time = random_double();

    // Returns the resulting ray from the origin and direction.
    return Ray(ray_origin, ray_direction, ray_time);
  }
#endif

  // Fallback scalar implementation.

  // This calculates the 3D world-space point that corresponds to pixel (i, j),
  // with subpixel jitter.
  Vec3 pixel_sample = m_pixel00_loc + ((i + offset.x()) * m_pixel_delta_u) +
                      ((j + offset.y()) * m_pixel_delta_v);

  // If defocus_angle == 0: use the camera center -> behaves like a pinhole
  // camera. Else: use a point sampled from a disk, simulating a lens with
  // radius and blur (depth of field).
  Point3 ray_origin = (m_defocus_angle <= 0) ? m_center : defocus_disk_sample();

  // Determine the ray direction by finding the vector difference between the
  // pixel sample and the ray origin.
  Vec3 ray_direction = pixel_sample - ray_origin;

  double ray_time = random_double();

  // Returns the resulting ray from the origin and direction.
  return Ray(ray_origin, ray_direction, ray_time);
}

Vec3 Camera::sample_square_stratified(int s_i, int s_j) const {
  int sqrt_spp = int(std::sqrt(m_samples_per_pixel));
  double recip_sqrt_spp = 1.0 / sqrt_spp;

  double px = ((s_i + random_double()) * recip_sqrt_spp) - 0.5;
  double py = ((s_j + random_double()) * recip_sqrt_spp) - 0.5;

  return Vec3(px, py, 0);
}

Vec3 Camera::sample_square() const {
  return Vec3(random_double() - 0.5, random_double() - 0.5, 0);
}

Vec3 Camera::sample_disk(double radius) const {
  return radius * random_in_unit_disk();
}

Point3 Camera::defocus_disk_sample() const {
  Point3 point = random_in_unit_disk();
  return m_center + (point[0] * m_defocus_disk_u) +
         (point[1] * m_defocus_disk_v);
}

Color Camera::ray_color(const Ray &ray, int depth, const HittableList &world,
                        const HittableList &lights) const {
  // Base case: if we've exceeded the ray bounce limit, no more light is
  // gathered.
  if (depth <= 0)
    return Color(0, 0, 0);

  HitRecord record;

  // If the ray hits nothing, return the background color.
  if (!world.hit(ray, Interval(0.001, INF), record))
    return m_background;

  // If the ray hits a hittable objct, we store the hit record and calculate the
  // scattering based on the material.
  ScatterRecord scatter_record;
  Color emitted_color =
      record.material->emitted(ray, record, record.u, record.v, record.point);

  // If the material does not scatter, we just return the material's emitted
  // color for that point.
  if (!record.material->scatter(ray, record, scatter_record))
    return emitted_color;

  // If the material scatters the ray in a specific direction without using a
  // PDF, follow that ray recursively and scale the result by the attenuation.
  // This ONLY happens when the material chooses to bypass the PDF sampling
  // logic.
  if (scatter_record.skip_pdf)
    return scatter_record.attenuation *
           ray_color(scatter_record.skip_pdf_ray, depth - 1, world, lights);

  // Create a PDF that samples directions toward the light sources from the hit
  // point. This helps concentrate rays toward bright areas for better rendering
  // efficiency.
  // NOTE: In the case that the lights hittable list object is empty (there's no
  // lights), we cannot use scattering from light sources, so we skip.
  PDFPtr light_ptr;
  if (lights.get_objects().empty())
    light_ptr = scatter_record.pdf;
  else
    light_ptr = std::make_shared<HittablePDF>(lights, record.point);

  // Combine two PDFs: one from the material's scattering function and one
  // toward the lights. This is a mixture of importance sampling strategies to
  // improve convergence.
  MixturePDF p(light_ptr, scatter_record.pdf);

  // Generate a new scattered ray based on the mixture PDF.
  // This ray originates from the hit point and is aimed in a sampled direction.
  Ray scattered = Ray(record.point, p.generate(), ray.time());

  // Compute the probability density function (PDF) value of the sampled
  // direction. This is used to correctly scale the Monte Carlo estimate.
  double pdf_value = p.value(scattered.direction());

  // Ask the material what its theoretical scattering PDF is for this
  // interaction. This is needed for physically-based importance sampling.
  double scattering_pdf =
      record.material->scattering_pdf(ray, record, scattered);

  // Recursively trace the scattered ray to get the color that arrives along it.
  // This captures indirect lighting and multiple bounces.
  Color sample_color = ray_color(scattered, depth - 1, world, lights);

  // Scale the returned color by:
  /// - `attenuation`: how much the material reduces the ray's energy.
  /// - `scattering_pdf`: how likely the material is to scatter in that
  /// direction.
  /// - `1 / pdf_value`: normalizes the estimate based on how likely we were to
  /// choose this ray.
  Color scattered_color =
      (scatter_record.attenuation * scattering_pdf * sample_color) / pdf_value;

  // Add any emitted light from the surface (like glowing surfaces) to the
  // scattered result.
  return emitted_color + scattered_color;
}