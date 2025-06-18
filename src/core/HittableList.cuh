#pragma once

#ifdef USE_CUDA

#include "../optimization/AABB.cuh"
#include "../utils/math/Interval.cuh"
#include "../utils/math/Utility.cuh"
#include "HitRecord.cuh"
#include "Hittable.cuh"
#include "Ray.cuh"

// Maximum objects per list for now (can be dynamic with memory allocator).
static const int MAX_HITTABLES_PER_LIST = 32;

// CUDA-compatible struct representing a list of hittables.
struct CudaHittableList {
  CudaHittable hittables[MAX_HITTABLES_PER_LIST];
  int count;
  CudaAABB bbox;

  __device__ CudaHittableList(CudaHittable *_hittables, int _num_hittables) {
    assert(_num_hittables <= MAX_HITTABLES_PER_LIST);

    count = _num_hittables;
    for (int i = 0; i < _num_hittables; i++) {
      hittables[i] = _hittables[i];
    }

    // Calculate bounding box from all objects.
    bbox = CudaAABB();
    for (int i = 0; i < count; i++) {
      bbox = CudaAABB(bbox, hittables[i].get_bounding_box());
    }
  }

  // Hit function: checks ray intersection against all hittables in the list.
  __device__ inline bool hit(const CudaRay &ray, CudaInterval t_range,
                             CudaHitRecord &out_rec, curandState *rand_state) {
    CudaHitRecord temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_range.max;

    for (int i = 0; i < count; i++) {
      if (hittables[i].hit(ray, CudaInterval(t_range.min, closest_so_far),
                           temp_rec, rand_state)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        out_rec = temp_rec;
      }
    }

    return hit_anything;
  }

  // PDF value is averaged over all sub-objects.
  __device__ inline double pdf_value(const CudaPoint3 &origin,
                                     const CudaVec3 &direction) {
    if (count == 0)
      return 0.0;

    double weight = 1.0 / count;
    double sum = 0.0;

    for (int i = 0; i < count; i++) {
      sum += weight * hittables[i].pdf_value(origin, direction);
    }

    return sum;
  }

  // Random direction toward one random object in the list.
  __device__ inline CudaVec3 random(const CudaPoint3 &origin,
                                    curandState *state) {
    // Pick random object from the list.
    int i = (int)(curand_uniform_double(state) * count);
    i = (i < 0) ? 0 : ((i >= count) ? count - 1 : i);
    return hittables[i].random(origin, state);
  }

  // Bounding box of the whole list.
  __device__ inline CudaAABB get_bounding_box() { return bbox; }
};

#endif // USE_CUDA