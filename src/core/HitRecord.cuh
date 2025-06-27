#pragma once

#ifdef USE_CUDA

#include "../utils/math/Vec3Utility.cuh"
#include "Ray.cuh"
#include "Vec3Types.cuh"
#include <iomanip>
#include <sstream>

// Forward declarations to avoid circular dependency.
struct CudaMaterial;

// Represents captured info from a ray hitting a hittable object in CUDA.
struct CudaHitRecord {
  // The point it hit at.
  CudaPoint3 point;

  // The normal vector to the ray/object intersection.
  CudaVec3 normal;

  // The material index of the hittable object.
  size_t material_index;

  // The parameter t (time) along the ray in which the ray hit the hittable
  // object.
  double t;

  // Front face = the ray hits the surface from outside (normal opposes the
  // ray). Back face = the ray hits from inside (normal faces same way as ray).
  bool front_face;

  // 2D coordinates used to map a 2D texture image onto a 3D surface. When a ray
  // hits a surface (like a sphere or triangle), the u and v values tell the
  // renderer which part of the texture to apply at that point. This would be
  // able to determine what the texture of the object looks like at the position
  // the ray hit the object.
  double u, v;
};

// Sets the hit record normal vector.
// NOTE: `outward_normal` is assumed to be unit length.
__device__ __forceinline__ void
cuda_hit_record_set_face_normal(CudaHitRecord &rec, const CudaRay &ray,
                                const CudaVec3 &outward_normal) {
  rec.front_face = cuda_vec3_dot_product(ray.direction, outward_normal) < 0.0;
  rec.normal =
      rec.front_face ? outward_normal : cuda_vec3_negate(outward_normal);
}

// JSON serialization function for CudaHitRecord.
inline std::string cuda_json_hit_record(const CudaHitRecord &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaHitRecord\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"point\":" << cuda_json_vec3(obj.point) << ",";
  oss << "\"normal\":" << cuda_json_vec3(obj.normal) << ",";
  oss << "\"material_index\":" << obj.material_index << ",";
  oss << "\"t\":" << obj.t << ",";
  oss << "\"front_face\":" << (obj.front_face ? "true" : "false") << ",";
  oss << "\"u\":" << obj.u << ",";
  oss << "\"v\":" << obj.v;
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA
