#pragma once

#include "../../utils/math/Vec3.hpp"
#include "../Vec3Types.hpp"

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
};