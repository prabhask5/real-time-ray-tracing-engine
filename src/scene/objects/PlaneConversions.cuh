#pragma once

#ifdef USE_CUDA

#include "../../optimization/AABBConversions.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "../materials/MaterialConversions.cuh"
#include "Plane.cuh"
#include "Plane.hpp"

// Convert CPU Plane to CUDA Plane.
inline CudaPlane cpu_to_cuda_plane(const Plane &cpu_plane) {
  // Extract plane properties using public getter methods.
  Point3 corner = cpu_plane.get_corner();
  Vec3 u_side = cpu_plane.get_u_side();
  Vec3 v_side = cpu_plane.get_v_side();
  MaterialPtr cpu_material = cpu_plane.get_material();

  // Convert properties to CUDA format.
  CudaPoint3 cuda_corner = cpu_to_cuda_vec3(corner);
  CudaVec3 cuda_u_side = cpu_to_cuda_vec3(u_side);
  CudaVec3 cuda_v_side = cpu_to_cuda_vec3(v_side);
  CudaMaterial cuda_material = cpu_to_cuda_material(*cpu_material);

  return CudaPlane(cuda_corner, cuda_u_side, cuda_v_side, cuda_material);
}

// Convert CUDA Plane to CPU Plane.
inline std::shared_ptr<Plane> cuda_to_cpu_plane(const CudaPlane &cuda_plane) {
  // Extract plane properties.
  Point3 cpu_corner = cuda_to_cpu_vec3(cuda_plane.corner);
  Vec3 cpu_u_side = cuda_to_cpu_vec3(cuda_plane.u_side);
  Vec3 cpu_v_side = cuda_to_cpu_vec3(cuda_plane.v_side);

  // Convert material.
  auto cpu_material = cuda_to_cpu_material(cuda_plane.material);

  // Create CPU plane.
  return std::make_shared<Plane>(cpu_corner, cpu_u_side, cpu_v_side,
                                 cpu_material);
}

// Helper function for creating planes with explicit parameters.
__host__ __device__ inline CudaPlane
create_cuda_plane(const Point3 &corner, const Vec3 &u_side, const Vec3 &v_side,
                  const CudaMaterial &material) {
  return CudaPlane(cpu_to_cuda_vec3(corner), cpu_to_cuda_vec3(u_side),
                   cpu_to_cuda_vec3(v_side), material);
}

// Helper function to create CUDA plane.
__host__ __device__ inline CudaPlane
create_cuda_plane(const CudaPoint3 &corner, const CudaVec3 &u_side,
                  const CudaVec3 &v_side, const CudaMaterial &material) {
  return CudaPlane(corner, u_side, v_side, material);
}

// Batch conversion functions.
void batch_cpu_to_cuda_plane(const Plane **cpu_planes,
                             const CudaMaterial *cuda_materials,
                             CudaPlane *cuda_planes, int count);
void batch_cuda_to_cpu_plane(const CudaPlane *cuda_planes,
                             std::shared_ptr<Plane> *cpu_planes, int count);

#endif // USE_CUDA