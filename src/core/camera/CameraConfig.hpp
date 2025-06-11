#pragma once

#include <Vec3.hpp>
#include <Vec3Types.hpp>

struct CameraConfig {
  double aspect_ratio = 1.0;
  int image_width = 600;
  int samples_per_pixel = 10;
  int max_depth = 10;
  Color background;

  double vfov = 90;
  Point3 lookfrom = Point3(0, 0, 0);
  Point3 lookat = Point3(0, 0, -1);
  Vec3 vup = Vec3(0, 1, 0);

  double defocus_angle = 0;
  double focus_dist = 10;

  bool use_parallelism = false;
  bool use_bvh = false;
  bool use_gpu = false;
};