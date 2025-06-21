#pragma once

#ifdef USE_CUDA

#include "../../core/RayConversions.cuh"
#include "../../optimization/AABBConversions.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "../materials/MaterialConversions.cuh"
#include "Sphere.cuh"
#include "Sphere.hpp"

// Convert CPU Sphere to CUDA Sphere.
inline CudaSphere cpu_to_cuda_sphere(const Sphere &cpu_sphere) {
  // Extract sphere properties using public getter methods.
  Point3 cpu_center = cpu_sphere.get_center();
  double radius = cpu_sphere.get_radius();
  MaterialPtr cpu_material = cpu_sphere.get_material();

  // Convert center and material to CUDA format.
  CudaPoint3 cuda_center = cpu_to_cuda_vec3(cpu_center);
  CudaMaterial cuda_material = cpu_to_cuda_material(*cpu_material);

  return CudaSphere(cuda_center, radius, cuda_material);
}

// Convert CPU Sphere with motion blur to CUDA Sphere.
__host__ __device__ inline CudaSphere
cpu_to_cuda_sphere_motion(const Sphere &cpu_sphere, const Point3 &before_center,
                          const Point3 &after_center, double radius,
                          const CudaMaterial &cuda_material) {
  return CudaSphere(cpu_to_cuda_vec3(before_center),
                    cpu_to_cuda_vec3(after_center), radius, cuda_material);
}

// Convert CUDA Sphere to CPU Sphere.
inline std::shared_ptr<Sphere>
cuda_to_cpu_sphere(const CudaSphere &cuda_sphere) {
  // Extract center at time 0.
  CudaPoint3 cuda_center = cuda_sphere.center.at(0.0);
  Point3 cpu_center = cuda_to_cpu_vec3(cuda_center);

  // Convert material.
  auto cpu_material = cuda_to_cpu_material(cuda_sphere.material);

  // Create CPU sphere.
  return std::make_shared<Sphere>(cpu_center, cuda_sphere.radius, cpu_material);
}

// Convert CUDA Sphere with motion blur to CPU Sphere.
inline std::shared_ptr<Sphere>
cuda_to_cpu_sphere_motion(const CudaSphere &cuda_sphere) {
  // Extract before and after centers.
  CudaPoint3 before_center = cuda_sphere.center.origin;
  CudaPoint3 after_center = cuda_sphere.center.at(1.0);

  Point3 cpu_before = cuda_to_cpu_vec3(before_center);
  Point3 cpu_after = cuda_to_cpu_vec3(after_center);

  // Convert material.
  auto cpu_material = cuda_to_cpu_material(cuda_sphere.material);

  // Create CPU sphere with motion blur.
  return std::make_shared<Sphere>(cpu_before, cpu_after, cuda_sphere.radius,
                                  cpu_material);
}

// Helper functions for creating spheres with explicit parameters.
__host__ __device__ inline CudaSphere
create_cuda_sphere_static(const Point3 &center, double radius,
                          const CudaMaterial &material) {
  return CudaSphere(cpu_to_cuda_vec3(center), radius, material);
}

__host__ __device__ inline CudaSphere
create_cuda_sphere_motion(const Point3 &before_center,
                          const Point3 &after_center, double radius,
                          const CudaMaterial &material) {
  return CudaSphere(cpu_to_cuda_vec3(before_center),
                    cpu_to_cuda_vec3(after_center), radius, material);
}

// Batch conversion functions.
void batch_cpu_to_cuda_sphere(const Sphere **cpu_spheres,
                              const CudaMaterial *cuda_materials,
                              CudaSphere *cuda_spheres, int count);
void batch_cuda_to_cpu_sphere(const CudaSphere *cuda_spheres,
                              std::shared_ptr<Sphere> *cpu_spheres, int count);

#endif // USE_CUDA