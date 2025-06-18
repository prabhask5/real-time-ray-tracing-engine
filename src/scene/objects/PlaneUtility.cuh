#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/HittableList.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../optimization/AABB.cuh"
#include "../../utils/math/Interval.cuh"
#include "../materials/Material.cuh"
#include "Plane.cuh"

// Adds six planes to the hittable list that form a box from `min` to `max` with
// given material.
__host__ __device__ inline void
cuda_make_box(CudaPlane *out_planes, // output array of 6 planes
              int base_index,        // starting index in out_planes
              const CudaPoint3 &a, const CudaPoint3 &b,
              const CudaMaterial &material) {

  CudaPoint3 min(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));

  CudaPoint3 max(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));

  CudaVec3 dx(max.x - min.x, 0, 0);
  CudaVec3 dy(0, max.y - min.y, 0);
  CudaVec3 dz(0, 0, max.z - min.z);

  int i = base_index;

  // +Z (front).
  out_planes[i++] =
      CudaPlane(CudaPoint3(min.x, min.y, max.z), dx, dy, material);

  // +X (right).
  out_planes[i++] =
      CudaPlane(CudaPoint3(max.x, min.y, max.z), -dz, dy, material);

  // -Z (back).
  out_planes[i++] =
      CudaPlane(CudaPoint3(max.x, min.y, min.z), -dx, dy, material);

  // -X (left).
  out_planes[i++] =
      CudaPlane(CudaPoint3(min.x, min.y, min.z), dz, dy, material);

  // +Y (top).
  out_planes[i++] =
      CudaPlane(CudaPoint3(min.x, max.y, max.z), dx, -dz, material);

  // -Y (bottom).
  out_planes[i++] =
      CudaPlane(CudaPoint3(min.x, min.y, min.z), dx, dz, material);
}

#endif // USE_CUDA
