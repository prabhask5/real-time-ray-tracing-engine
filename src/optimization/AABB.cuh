#pragma once

#ifdef USE_CUDA

#include "../core/Ray.cuh"
#include "../core/Vec3Types.cuh"
#include "../utils/math/Interval.cuh"

// Axis-Aligned Bounding Box in CUDA: tight POD layout, GPU-friendly.
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

  __device__ __forceinline__ bool hit(const CudaRay &ray,
                                      CudaInterval t_values) {
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

  __device__ __forceinline__ void pad_to_minimums(double delta = 0.0001) {
    if (x.size() < delta)
      x = x.expand(delta);
    if (y.size() < delta)
      y = y.expand(delta);
    if (z.size() < delta)
      z = z.expand(delta);
  }
};

__device__ __constant__ CudaAABB CUDA_EMPTY_AABB = {
    CUDA_EMPTY_INTERVAL, CUDA_EMPTY_INTERVAL, CUDA_EMPTY_INTERVAL};
__device__ __constant__ CudaAABB CUDA_UNIVERSE_AABB = {
    CUDA_UNIVERSE_INTERVAL, CUDA_UNIVERSE_INTERVAL, CUDA_UNIVERSE_INTERVAL};

#endif // USE_CUDA
