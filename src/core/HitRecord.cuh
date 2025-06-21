#pragma once

#ifdef USE_CUDA

#include "../utils/math/Vec3Utility.cuh"
#include "Ray.cuh"
#include "Vec3Types.cuh"

// Forward declarations to avoid circular dependency.
struct CudaMaterial;

// Represents captured info from a ray hitting a hittable object in CUDA.
struct CudaHitRecord {
  // The point it hit at.
  CudaPoint3 point;

  // The normal vector to the ray/object intersection.
  CudaVec3 normal;

  // The material of the hittable object.
  CudaMaterial *material_pointer;

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
cuda_set_face_normal(CudaHitRecord &rec, const CudaRay &ray,
                     const CudaVec3 &outward_normal) {
  rec.front_face = cuda_dot_product(ray.direction, outward_normal) < 0.0;
  rec.normal = rec.front_face ? outward_normal : -outward_normal;
}

#endif // USE_CUDA
