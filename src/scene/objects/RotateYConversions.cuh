#pragma once

#ifdef USE_CUDA

#include "../../optimization/AABBConversions.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "RotateY.cuh"

// Forward declarations.
class RotateY;
struct CudaHittable;
class Hittable;

// Forward declaration of conversion functions.
inline CudaHittable cpu_to_cuda_hittable(const HittablePtr &cpu_hittable);
inline HittablePtr cuda_to_cpu_hittable(const CudaHittable &cuda_hittable);

// Convert CPU RotateY to CUDA RotateY.
inline CudaRotateY cpu_to_cuda_rotate_y(const RotateY &cpu_rotate_y) {
  // Extract properties from CPU RotateY using getters.
  auto cpu_object = cpu_rotate_y.get_object();
  double angle_degrees = cpu_rotate_y.get_angle();
  AABB cpu_bbox = cpu_rotate_y.get_bounding_box();

  // Convert object to CUDA format.
  CudaHittable *cuda_object = new CudaHittable();
  *cuda_object = cpu_to_cuda_hittable(cpu_object);

  // Convert bounding box.
  CudaAABB cuda_bbox = cpu_to_cuda_aabb(cpu_bbox);

  // Create CUDA RotateY.
  return CudaRotateY(cuda_object, angle_degrees, cuda_bbox);
}

// Convert CUDA RotateY to CPU RotateY.
inline RotateY cuda_to_cpu_rotate_y(const CudaRotateY &cuda_rotate_y) {
  // Convert object back to CPU format.
  auto cpu_object = cuda_to_cpu_hittable(*cuda_rotate_y.object);

  // Calculate angle from sin/cos values.
  double angle_degrees =
      atan2(cuda_rotate_y.sin_theta, cuda_rotate_y.cos_theta) * 180.0 / M_PI;

  // Create CPU RotateY.
  return RotateY(cpu_object, angle_degrees);
}

// Create CUDA RotateY from hittable and angle.
__host__ __device__ inline CudaRotateY
create_cuda_rotate_y(const CudaHittable *object, double angle_degrees) {
  // Get original bounding box.
  CudaAABB original_bbox = object->get_bounding_box();

  return CudaRotateY(object, angle_degrees, original_bbox);
}

// Helper function to create rotated sphere.
__host__ __device__ inline CudaRotateY
create_cuda_rotated_sphere(const CudaPoint3 &center, double radius,
                           const CudaMaterial &material, double angle_degrees) {
  CudaHittable *sphere_obj = new CudaHittable();
  sphere_obj->type = HITTABLE_SPHERE;
  sphere_obj->sphere = create_cuda_sphere_static(center, radius, material);

  CudaAABB sphere_bbox = sphere_obj->get_bounding_box();
  return CudaRotateY(sphere_obj, angle_degrees, sphere_bbox);
}

// Helper function to create rotated plane.
__host__ __device__ inline CudaRotateY
create_cuda_rotated_plane(const CudaPoint3 &corner, const CudaVec3 &u,
                          const CudaVec3 &v, const CudaMaterial &material,
                          double angle_degrees) {
  CudaHittable *plane_obj = new CudaHittable();
  plane_obj->type = HITTABLE_PLANE;
  plane_obj->plane = create_cuda_plane(corner, u, v, material);

  CudaAABB plane_bbox = plane_obj->get_bounding_box();
  return CudaRotateY(plane_obj, angle_degrees, plane_bbox);
}

// Memory management for rotate objects.
inline void cleanup_cuda_rotate_y(CudaRotateY &cuda_rotate_y) {
  if (cuda_rotate_y.object) {
    delete cuda_rotate_y.object;
    cuda_rotate_y.object = nullptr;
  }
}

#endif // USE_CUDA