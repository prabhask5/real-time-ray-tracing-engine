#pragma once

#ifdef USE_CUDA

#include "../core/Ray.cuh"
#include "../core/Vec3Types.hpp"
#include "../utils/math/Interval.cuh"

// Axis-Aligned Bounding Box in CUDA: tight POD layout, GPU-friendly.
struct CudaAABB {
  CudaInterval x;
  CudaInterval y;
  CudaInterval z;
};

__device__ __forceinline__ CudaAABB cuda_make_aabb(const CudaInterval &x,
                                                   const CudaInterval &y,
                                                   const CudaInterval &z) {
  CudaAABB box = {x, y, z};
  return box;
}

__device__ __forceinline__ CudaAABB
cuda_make_aabb_from_points(const CudaPoint3 &p1, const CudaPoint3 &p2) {
  CudaAABB box;
  box.x = cuda_interval_minmax(p1.x, p2.x);
  box.y = cuda_interval_minmax(p1.y, p2.y);
  box.z = cuda_interval_minmax(p1.z, p2.z);
  return box;
}

__device__ __forceinline__ CudaAABB cuda_surrounding_aabb(const CudaAABB &a,
                                                          const CudaAABB &b) {
  CudaAABB box;
  box.x = cuda_surrounding_interval(a.x, b.x);
  box.y = cuda_surrounding_interval(a.y, b.y);
  box.z = cuda_surrounding_interval(a.z, b.z);
  return box;
}

__device__ __forceinline__ int cuda_aabb_longest_axis(const CudaAABB &box) {
  double x_size = cuda_interval_size(box.x);
  double y_size = cuda_interval_size(box.y);
  double z_size = cuda_interval_size(box.z);

  if (x_size > y_size && x_size > z_size)
    return 0;
  if (y_size > z_size)
    return 1;
  return 2;
}

__device__ __forceinline__ bool
cuda_aabb_hit(const CudaAABB &box, const CudaRay &ray, CudaInterval t_values) {
  for (int axis = 0; axis < 3; ++axis) {
    const CudaInterval &ax = (axis == 0) ? box.x : (axis == 1) ? box.y : box.z;
    double origin_axis = ray.origin[axis];
    double direction_axis = ray.direction[axis];
    double inv_d = 1.0 / direction_axis;

    double t0 = (ax.min - origin_axis) * inv_d;
    double t1 = (ax.max - origin_axis) * inv_d;

    if (inv_d < 0.0) {
      double temp = t0;
      t0 = t1;
      t1 = temp;
    }

    t_values.min = t0 > t_values.min ? t0 : t_values.min;
    t_values.max = t1 < t_values.max ? t1 : t_values.max;
    if (t_values.max <= t_values.min)
      return false;
  }
  return true;
}

__device__ __forceinline__ CudaAABB cuda_aabb_padded(const CudaAABB &box,
                                                     double delta = 0.0001) {
  CudaAABB result = box;
  if (cuda_interval_size(result.x) < delta)
    result.x = cuda_interval_expand(result.x, delta);
  if (cuda_interval_size(result.y) < delta)
    result.y = cuda_interval_expand(result.y, delta);
  if (cuda_interval_size(result.z) < delta)
    result.z = cuda_interval_expand(result.z, delta);
  return result;
}

__device__ __constant__ CudaAABB CudaEmptyAABB = {
    CudaEmptyInterval, CudaEmptyInterval, CudaEmptyInterval};
__device__ __constant__ CudaAABB CudaUniverseAABB = {
    CudaUniverseInterval, CudaUniverseInterval, CudaUniverseInterval};

#endif // USE_CUDA
