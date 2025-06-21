#pragma once

#ifdef USE_CUDA

#include "../../optimization/AABBConversions.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "RotateY.cuh"
#include "RotateY.hpp"
#include "SphereConversions.cuh"

// Forward declaration for hittable conversion.
struct CudaHittable;
CudaHittable cpu_to_cuda_hittable(const Hittable &cpu_hittable);
HittablePtr cuda_to_cpu_hittable(const CudaHittable &cuda_hittable);

// Convert CPU RotateY to CUDA RotateY.
inline CudaRotateY cpu_to_cuda_rotate_y(const RotateY &cpu_rotate_y) {
  // Extract properties from CPU RotateY using getters.
  HittablePtr cpu_object = cpu_rotate_y.get_object();
  double angle_degrees = cpu_rotate_y.get_angle();
  AABB cpu_bbox = cpu_rotate_y.get_bounding_box();

  // Convert object to CUDA format.
  CudaHittable *cuda_object = new CudaHittable();
  *cuda_object = cpu_to_cuda_hittable(*cpu_object);

  // Create CUDA RotateY.
  return CudaRotateY(cuda_object, angle_degrees);
}

// Convert CUDA RotateY to CPU RotateY.
inline RotateY cuda_to_cpu_rotate_y(const CudaRotateY &cuda_rotate_y) {
  // Convert object back to CPU format.
  HittablePtr cpu_object = cuda_to_cpu_hittable(*cuda_rotate_y.object);

  // Calculate angle from sin/cos values.
  double angle_degrees =
      atan2(cuda_rotate_y.sin_theta, cuda_rotate_y.cos_theta) * 180.0 / M_PI;

  // Create CPU RotateY.
  return RotateY(cpu_object, angle_degrees);
}

// Memory management for rotate objects.
inline void cleanup_cuda_rotate_y(CudaRotateY &cuda_rotate_y) {
  if (cuda_rotate_y.object) {
    delete cuda_rotate_y.object;
    cuda_rotate_y.object = nullptr;
  }
}

#endif // USE_CUDA