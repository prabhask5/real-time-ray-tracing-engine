#pragma once

#ifdef USE_CUDA

#include "../../optimization/AABBConversions.cuh"
#include "../../utils/math/Utility.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "RotateY.cuh"
#include "RotateY.hpp"
#include <cmath>

// Forward declaration.
struct CudaHittable;
CudaHittable cpu_to_cuda_hittable(const Hittable &cpu_hittable);

// Convert CPU RotateY to CUDA RotateY.
inline CudaRotateY cpu_to_cuda_rotate_y(const RotateY &cpu_rotate_y) {
  // Extract properties from CPU RotateY using getters.
  HittablePtr cpu_object = cpu_rotate_y.get_object();
  double angle_degrees = cpu_rotate_y.get_angle();
  AABB cpu_bbox = cpu_rotate_y.get_bounding_box();

  CudaHittable *cuda_object = new CudaHittable();
  *cuda_object = cpu_to_cuda_hittable(*cpu_object);

  double radians = angle_degrees * CUDA_PI / 180.0;
  return cuda_make_rotate_y(cuda_object, sin(radians), cos(radians),
                            cpu_to_cuda_aabb(cpu_bbox));
}

#endif // USE_CUDA