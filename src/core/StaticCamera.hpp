#pragma once

#include "CameraConfig.hpp"
#include <Vec3.hpp>
#include <Vec3Types.hpp>
#include <string>

class Hittable; // From Hittable.hpp.
class Ray;      // From Ray.hpp.

class StaticCamera {
public:
  StaticCamera(const CameraConfig &config, std::string output_file_name);

  // The main rendering loop:
  // - Calls initialize() to set up internal camera vectors.
  // - Outputs a header for the PPM image format.
  // - Loops over each pixel row-by-row.
  // - For each pixel:
  // - - Shoots samples_per_pixel rays with slight jitter.
  // - - Averages their returned colors.
  // - - Writes the result using write_color(...).
  void render(const Hittable &world, const Hittable &lights);

private:
  // Internal metadata.

  // Rendered image height.
  int m_image_height;

  // Color scale factor for a sum of pixel samples.
  double m_pixel_samples_scale;

  // Camera center.
  Point3 m_center;

  // Location of pixel 0, 0.
  Point3 m_pixel00_loc;

  // Offset to pixel to the right.
  Vec3 m_pixel_delta_u;

  // Offset to pixel below.
  Vec3 m_pixel_delta_v;

  // Camera frame basis vectors.
  Vec3 m_u, m_v, m_w;

  // Defocus disk horizontal radius.
  Vec3 m_defocus_disk_u;

  // Defocus disk vertical radius.
  Vec3 m_defocus_disk_v;

  // Camera config metadata.

  // Output image file name.
  std::string m_output_file;

  // Ratio of image width over height.
  double m_aspect_ratio;

  // Rendered image width in pixel count.
  int m_image_width = 100;

  // Count of random samples for each pixel.
  int m_samples_per_pixel = 10;

  // Maximum number of ray bounces into scene.
  int m_max_depth = 10;

  // Scene background color.
  Color m_background;

  // Vertical view angle (field of view).
  double m_vfov = 90;

  // Point camera is looking from.
  Point3 m_lookfrom = Point3(0, 0, 0);

  // Point camera is looking at.
  Point3 m_lookat = Point3(0, 0, -1);

  // Camera-relative "up" direction.
  Vec3 m_vup = Vec3(0, 1, 0);

  // Variation angle of rays through each pixel.
  double m_defocus_angle = 0;

  // Distance from camera lookfrom point to plane of perfect focus.
  double m_focus_dist = 10;

  // Calculates all internal fields used to construct rays.
  void initialize();

  // Construct a camera ray originating from the defocus disk and directed at a
  // randomly sampled point around the pixel location i, j for stratified sample
  // square s_i, s_j.
  Ray get_ray(int i, int j, int s_i, int s_j) const;

  // Returns the vector to a random point in the square sub-pixel specified by
  // grid indices s_i and s_j, for an idealized unit square pixel [-.5,-.5] to
  // [+.5,+.5].
  Vec3 sample_square_stratified(int s_i, int s_j) const;

  // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit
  // square.
  Vec3 sample_square() const;

  // Returns a random point in the unit (radius 0.5) disk centered at the
  // origin.
  Vec3 sample_disk(double radius) const;

  // Returns a random point in the camera defocus disk.
  Point3 defocus_disk_sample() const;

  // Recursively computes the color of a ray path:
  // - If the ray hits something:
  // - - Calls scatter(...) on the material.
  // - - Multiplies by attenuation and continues recursion.
  // - If it hits nothing:
  // - - Returns a gradient sky color.
  Color ray_color(const Ray &ray, int depth, const Hittable &world,
                  const Hittable &lights) const;
};