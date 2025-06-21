#pragma once

#ifdef USE_CUDA

#include "../core/Ray.cuh"
#include "../core/Vec3Types.cuh"
#include "../utils/math/Interval.cuh"

// Defines an Axis-Aligned Bounding Box: the simplest type of bounding box: a
// box aligned with the coordinate axes, defined by its minimum and maximum
// points in 3D space. Checking ray-box intersections is much faster than
// ray-triangle or ray-sphere intersections. So you wrap objects in AABBs and
// check the ray against those first.
struct CudaAABB {
  CudaInterval x;
  CudaInterval y;
  CudaInterval z;

  __device__ CudaAABB() {}

  __device__ CudaAABB(const CudaInterval &_x, const CudaInterval &_y,
                      const CudaInterval &_z)
      : x(_x), y(_y), z(_z) {
    pad_to_minimums();
  }

  __device__ CudaAABB(const CudaPoint3 &p1, const CudaPoint3 &p2) {
    x = CudaInterval(fmin(p1.x, p2.x), fmax(p1.x, p2.x));
    y = CudaInterval(fmin(p1.y, p2.y), fmax(p1.y, p2.y));
    z = CudaInterval(fmin(p1.z, p2.z), fmax(p1.z, p2.z));
    pad_to_minimums();
  }

  __device__ CudaAABB(const CudaAABB &a, const CudaAABB &b) {
    x = CudaInterval(a.x, b.x);
    y = CudaInterval(a.y, b.y);
    z = CudaInterval(a.z, b.z);
  }

  __device__ __forceinline__ const CudaInterval &
  get_axis_interval(int index) const {
    if (index == 1)
      return y;
    if (index == 2)
      return z;
    return x;
  }

  __device__ __forceinline__ int get_longest_axis() const {
    double x_size = x.size();
    double y_size = y.size();
    double z_size = z.size();

    if (x_size > y_size && x_size > z_size)
      return 0;
    if (y_size > z_size)
      return 1;
    return 2;
  }

  // This function checks if the ray hits the hittable object with the t values
  // in the interval range ray_t.
  __device__ __forceinline__ bool hit(const CudaRay &ray,
                                      CudaInterval t_values) const {
    for (int axis = 0; axis < 3; ++axis) {
      const CudaInterval &ax = get_axis_interval(axis);
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

  // Adjust the AABB so that no side is narrower than some delta, padding if
  // necessary.
  __device__ __forceinline__ void pad_to_minimums(double delta = 0.0001) {
    if (x.size() < delta)
      x = x.expand(delta);
    if (y.size() < delta)
      y = y.expand(delta);
    if (z.size() < delta)
      z = z.expand(delta);
  }
};

// Constants — use inline functions to avoid initialization issues
__device__ inline CudaAABB cuda_empty_aabb() {
  return CudaAABB(CUDA_EMPTY_INTERVAL, CUDA_EMPTY_INTERVAL,
                  CUDA_EMPTY_INTERVAL);
}

__device__ inline CudaAABB cuda_universe_aabb() {
  return CudaAABB(CUDA_UNIVERSE_INTERVAL, CUDA_UNIVERSE_INTERVAL,
                  CUDA_UNIVERSE_INTERVAL);
}

// For compatibility, define macros
#define CUDA_EMPTY_AABB cuda_empty_aabb()
#define CUDA_UNIVERSE_AABB cuda_universe_aabb()

#endif // USE_CUDA
