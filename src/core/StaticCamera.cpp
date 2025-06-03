#include "StaticCamera.hpp"
#include "HitRecord.hpp"
#include "Material.hpp"
#include "Ray.hpp"
#include <ColorUtility.hpp>
#include <Hittable.hpp>
#include <Vec3Utility.hpp>
#include <iostream>

StaticCamera::StaticCamera(const StaticCameraConfig &config)
    : m_aspect_ratio(config.aspect_ratio), m_image_width(config.image_width),
      m_samples_per_pixel(config.samples_per_pixel),
      m_max_depth(config.max_depth), m_vfov(config.vfov),
      m_lookfrom(config.lookfrom), m_lookat(config.lookat), m_vup(config.vup),
      m_defocus_angle(config.defocus_angle), m_focus_dist(config.focus_dist) {}

void StaticCamera::render(const Hittable &world) {
  initialize();

  // Writes to .ppm file by redirecting.
  std::cout << "P3\n" << m_image_width << ' ' << m_image_height << "\n255\n";

  for (int j = 0; j < m_image_height; j++) {
    std::clog << "\rScanlines remaining: " << (m_image_height - j) << ' '
              << std::flush;
    for (int i = 0; i < m_image_width; i++) {
      Color pixel_color(0, 0, 0);

      // Shoots samples_per_pixel rays per pixel.
      for (int sample = 0; sample < m_samples_per_pixel; sample++) {
        // Beams a ray from the defocus disk at the camera's position to where
        // near the pixel to simulate the light scattering. Slightly jitters
        // each ray inside the pixel, helps with anti-aliasing.
        Ray ray = get_ray(i, j);

        // Accumulates the color returned from each ray using recursive ray
        // tracing.
        pixel_color += ray_color(ray, m_max_depth, world);
      }

      // Normalizes the summed pixel colors (averages by the number of samples)
      // and converts to 8-bit [0,255] RGB and writes to standard output.
      write_color(std::cout, m_pixel_samples_scale * pixel_color);
    }
  }

  std::clog << "\rDone.                 \n";
}

void StaticCamera::initialize() {
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

Ray StaticCamera::get_ray(int i, int j) const {
  // Here we are getting a random vector within the pixel square, this makes
  // sure each ray goes through a different point in the pixel, prevents
  // aliasing. If this was not included, every ray would go through the center
  // of the pixel.
  Vec3 offset = sample_square();

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

  // Returns the resulting ray from the origin and direction.
  return Ray(ray_origin, ray_direction);
}

Vec3 StaticCamera::sample_square() const {
  return Vec3(random_double() - 0.5, random_double() - 0.5, 0);
}

Vec3 StaticCamera::sample_disk(double radius) const {
  return radius * random_in_unit_disk();
}

Point3 StaticCamera::defocus_disk_sample() const {
  Point3 point = random_in_unit_disk();
  return m_center + (point[0] * m_defocus_disk_u) +
         (point[1] * m_defocus_disk_v);
}

Color StaticCamera::ray_color(const Ray &r, int depth,
                              const Hittable &world) const {
  // Base case: if we've exceeded the ray bounce limit, no more light is
  // gathered.
  if (depth <= 0)
    return Color(0, 0, 0);

  HitRecord record;

  // If the ray hits a hittable objct, we store the hit record and calculate the
  // scattering basec on the material.
  if (world.hit(r, Interval(0.001, INF), record)) {
    Ray scattered;
    Color attenuation;

    // ONLY in the case that the material does not absorb the ray (scatter ret
    // is true), we proceed with increased depth.
    if (record.material->scatter(r, record, attenuation, scattered))
      return attenuation * ray_color(scattered, depth - 1, world);

    // If the material absorbs the ray, no more light is gathered.
    return Color(0, 0, 0);
  }

  // Generates a simple sky gradient for rays that miss all objects in the scene
  // — it's the background color of the scene.
  Vec3 unit_direction = unit_vector(r.direction());
  double a =
      0.5 *
      (unit_direction.y() +
       1.0); // Determines the gradient between blue and white to show the sky.
  return (1.0 - a) * Color(1.0, 1.0, 1.0) +
         a * Color(0.5, 0.7,
                   1.0); // We can see this very clearly here, the first color
                         // is white, and the second is blue.
}