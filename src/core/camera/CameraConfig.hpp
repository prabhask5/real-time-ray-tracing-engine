#pragma once

#include "../../utils/math/Vec3.hpp"
#include "../Vec3Types.hpp"
#include <iomanip>
#include <sstream>

// Memory layout optimized for camera configuration access.
struct alignas(16) CameraConfig {
  // Hot data: most frequently accessed during rendering setup.

  int image_width = 600;      // Image width (checked constantly)
  int samples_per_pixel = 10; // Sample count (checked per pixel)
  int max_depth = 10;         // Ray depth limit (checked per ray)

  // Group doubles together for optimal packing and SIMD potential.

  double aspect_ratio = 1.0; // Width/height ratio.
  double vfov = 90;          // Vertical field of view.
  double defocus_angle = 0;  // Aperture size for depth of field.
  double focus_dist = 10;    // Distance to focus plane.

  // Warm data: Vec3 types grouped together (16-byte aligned already).

  Point3 lookfrom = Point3(0, 0, 0); // Camera position.
  Point3 lookat = Point3(0, 0, -1);  // Look at point.
  Vec3 vup = Vec3(0, 1, 0);          // Up vector.
  Color background = Color(0, 0, 0); // Background color.

  // Cold data: boolean flags grouped together.

  bool use_parallelism = false; // Enable parallel processing.
  bool use_bvh = false;         // Enable BVH acceleration.
  bool use_gpu = false;         // Enable GPU rendering.
  bool use_debug = false;       // Enable debug mode for more logging.

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"CameraConfig\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"image_width\":" << image_width << ",";
    oss << "\"samples_per_pixel\":" << samples_per_pixel << ",";
    oss << "\"max_depth\":" << max_depth << ",";
    oss << "\"aspect_ratio\":" << aspect_ratio << ",";
    oss << "\"vfov\":" << vfov << ",";
    oss << "\"defocus_angle\":" << defocus_angle << ",";
    oss << "\"focus_dist\":" << focus_dist << ",";
    oss << "\"lookfrom\":" << lookfrom.json() << ",";
    oss << "\"lookat\":" << lookat.json() << ",";
    oss << "\"vup\":" << vup.json() << ",";
    oss << "\"background\":" << background.json() << ",";
    oss << "\"use_parallelism\":" << (use_parallelism ? "true" : "false")
        << ",";
    oss << "\"use_bvh\":" << (use_bvh ? "true" : "false") << ",";
    oss << "\"use_debug\":" << (use_debug ? "true" : "false") << ",";
    oss << "\"use_gpu\":" << (use_gpu ? "true" : "false");
    oss << "}";
    return oss.str();
  }
};