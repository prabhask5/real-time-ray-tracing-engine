#pragma once

#ifdef USE_CUDA

#include "../../optimization/AABBConversions.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "../materials/MaterialConversions.cuh"
#include "Plane.cuh"
#include "Plane.hpp"

// Convert CPU Plane to CUDA Plane POD struct.
inline CudaPlane cpu_to_cuda_plane(const Plane &cpu_plane) {
  Point3 corner = cpu_plane.get_corner();
  Vec3 u_side = cpu_plane.get_u_side();
  Vec3 v_side = cpu_plane.get_v_side();

  CudaPoint3 cuda_corner = cpu_to_cuda_vec3(corner);
  CudaVec3 cuda_u_side = cpu_to_cuda_vec3(u_side);
  CudaVec3 cuda_v_side = cpu_to_cuda_vec3(v_side);

  // Leave material blank for now.
  return cuda_make_plane(cuda_corner, cuda_u_side, cuda_v_side, nullptr);
}

#endif // USE_CUDA