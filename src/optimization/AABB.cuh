#pragma once

#ifdef USE_CUDA

#include "../core/Ray.cuh"
#include "../core/Vec3Types.cuh"
#include "../utils/math/Interval.cuh"

// POD struct representing an Axis-Aligned Bounding Box.
struct CudaAABB {
  CudaInterval x;
  CudaInterval y;
  CudaInterval z;
};

// Forward declaration.
__host__ __device__ __forceinline__ void
cuda_aabb_pad_to_minimums(CudaAABB &aabb, double delta = 0.0001);

// AABB initialization functions.
__host__ __device__ inline CudaAABB cuda_make_aabb() {
  CudaAABB aabb;
  aabb.x = cuda_make_interval();
  aabb.y = cuda_make_interval();
  aabb.z = cuda_make_interval();
  return aabb;
}

__host__ __device__ inline CudaAABB cuda_make_aabb(const CudaInterval &x,
                                                   const CudaInterval &y,
                                                   const CudaInterval &z) {
  CudaAABB aabb;
  aabb.x = x;
  aabb.y = y;
  aabb.z = z;
  cuda_aabb_pad_to_minimums(aabb);
  return aabb;
}

__host__ __device__ inline CudaAABB cuda_make_aabb(const CudaPoint3 &p1,
                                                   const CudaPoint3 &p2) {
  CudaAABB aabb;
  aabb.x = cuda_make_interval(fmin(p1.x, p2.x), fmax(p1.x, p2.x));
  aabb.y = cuda_make_interval(fmin(p1.y, p2.y), fmax(p1.y, p2.y));
  aabb.z = cuda_make_interval(fmin(p1.z, p2.z), fmax(p1.z, p2.z));
  cuda_aabb_pad_to_minimums(aabb);
  return aabb;
}

__host__ __device__ inline CudaAABB cuda_make_aabb(const CudaAABB &a,
                                                   const CudaAABB &b) {
  CudaAABB aabb;
  aabb.x = cuda_make_interval(a.x, b.x);
  aabb.y = cuda_make_interval(a.y, b.y);
  aabb.z = cuda_make_interval(a.z, b.z);
  return aabb;
}

// AABB utility functions.
__device__ __forceinline__ const CudaInterval &
cuda_aabb_get_axis_interval(const CudaAABB &aabb, int index) {
  if (index == 1)
    return aabb.y;
  if (index == 2)
    return aabb.z;
  return aabb.x;
}

__device__ __forceinline__ int
cuda_aabb_get_longest_axis(const CudaAABB &aabb) {
  double x_size = cuda_interval_size(aabb.x);
  double y_size = cuda_interval_size(aabb.y);
  double z_size = cuda_interval_size(aabb.z);

  if (x_size > y_size && x_size > z_size)
    return 0;
  if (y_size > z_size)
    return 1;
  return 2;
}

// This function checks if the ray hits the hittable object with the t values
// in the interval range ray_t.
__device__ __forceinline__ bool
cuda_aabb_hit(const CudaAABB &aabb, const CudaRay &ray, CudaInterval t_values) {
  for (int axis = 0; axis < 3; ++axis) {
    const CudaInterval &ax = cuda_aabb_get_axis_interval(aabb, axis);
    double origin_axis = cuda_vec3_get(ray.origin, axis);
    double direction_axis = cuda_vec3_get(ray.direction, axis);
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

// Adjust the AABB so that no side is narrower than some delta, padding if
// necessary.
__host__ __device__ __forceinline__ void
cuda_aabb_pad_to_minimums(CudaAABB &aabb, double delta) {
  if (cuda_interval_size(aabb.x) < delta)
    aabb.x = cuda_interval_expand(aabb.x, delta);
  if (cuda_interval_size(aabb.y) < delta)
    aabb.y = cuda_interval_expand(aabb.y, delta);
  if (cuda_interval_size(aabb.z) < delta)
    aabb.z = cuda_interval_expand(aabb.z, delta);
}

// Constants — use inline functions to avoid initialization issues.
__device__ inline CudaAABB cuda_empty_aabb() {
  return cuda_make_aabb(CUDA_EMPTY_INTERVAL, CUDA_EMPTY_INTERVAL,
                        CUDA_EMPTY_INTERVAL);
}

__device__ inline CudaAABB cuda_universe_aabb() {
  return cuda_make_aabb(CUDA_UNIVERSE_INTERVAL, CUDA_UNIVERSE_INTERVAL,
                        CUDA_UNIVERSE_INTERVAL);
}

// For compatibility, define macros.
#define CUDA_EMPTY_AABB cuda_empty_aabb()
#define CUDA_UNIVERSE_AABB cuda_universe_aabb()

#endif // USE_CUDA
