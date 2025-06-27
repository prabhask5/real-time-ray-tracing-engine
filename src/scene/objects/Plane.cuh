#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../optimization/AABB.cuh"
#include "../../utils/math/Interval.cuh"
#include "../../utils/math/Vec3Utility.cuh"
#include "../materials/Material.cuh"
#include <curand_kernel.h>
#include <iomanip>
#include <sstream>

// POD struct representing a planar parallelogram.
struct CudaPlane {
  CudaPoint3 corner;
  CudaVec3 u_side;
  CudaVec3 v_side;
  CudaVec3 w; // Used for computing UV coordinates.
  CudaVec3 normal;
  double D;
  double surface_area;
  CudaAABB bbox;
  size_t material_index;
};

// Plane initialization functions.
CudaPlane cuda_make_plane(const CudaPoint3 &corner, const CudaVec3 &u_side,
                          const CudaVec3 &v_side, size_t material_index,
                          const CudaAABB &bbox);

// Plane utility functions.
__device__ bool cuda_plane_hit(const CudaPlane &plane, const CudaRay &ray,
                               CudaInterval t_values, CudaHitRecord &record,
                               curandState *rand_state);

__device__ double cuda_plane_pdf_value(const CudaPlane &plane,
                                       const CudaPoint3 &origin,
                                       const CudaVec3 &direction);

__device__ CudaVec3 cuda_plane_random(const CudaPlane &plane,
                                      const CudaPoint3 &origin,
                                      curandState *rand_state);

__host__ __device__ inline CudaAABB
cuda_plane_get_bounding_box(const CudaPlane &plane) {
  return plane.bbox;
}

// JSON serialization function for CudaPlane.
inline std::string cuda_json_plane(const CudaPlane &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaPlane\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"corner\":" << cuda_json_vec3(obj.corner) << ",";
  oss << "\"u_side\":" << cuda_json_vec3(obj.u_side) << ",";
  oss << "\"v_side\":" << cuda_json_vec3(obj.v_side) << ",";
  oss << "\"w\":" << cuda_json_vec3(obj.w) << ",";
  oss << "\"normal\":" << cuda_json_vec3(obj.normal) << ",";
  oss << "\"D\":" << obj.D << ",";
  oss << "\"surface_area\":" << obj.surface_area << ",";
  oss << "\"bbox\":" << cuda_json_aabb(obj.bbox) << ",";
  oss << "\"material_index\":" << obj.material_index;
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA
