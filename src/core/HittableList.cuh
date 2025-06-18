#pragma once

#ifdef USE_CUDA

#include "AABB.cuh"
#include "HitRecord.cuh"
#include "Hittable.cuh"
#include "Interval.cuh"
#include "Ray.cuh"
#include "Utility.cuh"

// Max objects per list for now (can be dynamic with memory allocator)
static const int MAX_HITTABLES_PER_LIST = 32;

// CUDA-compatible struct representing a list of hittables.
struct CudaHittableListData {
  CudaHittable hittables[MAX_HITTABLES_PER_LIST];
  int count;
  CudaAABB bbox;
};

// Computes AABB from all hittables in the list.
__device__ inline CudaAABB cuda_hittable_list_bbox(CudaHittableListData *list) {
  CudaAABB box = list->bbox;
  for (int i = 0; i < list->count; i++) {
    box = cuda_surrounding_aabb(box,
                                cuda_hittable_bounding_box(list->hittables[i]));
  }
  return box;
}

// Hit function: checks ray intersection against all hittables in the list.
__device__ inline bool cuda_hittable_list_hit(const void *ptr,
                                              const CudaRay &ray,
                                              CudaInterval t_range,
                                              CudaHitRecord &out_rec) {
  const auto *list = static_cast<const CudaHittableListData *>(ptr);
  CudaHitRecord temp_rec;
  bool hit_anything = false;
  double closest_so_far = t_range.max;

  for (int i = 0; i < list->count; i++) {
    if (cuda_hittable_hit(list->hittables[i], ray,
                          CudaInterval(t_range.min, closest_so_far),
                          temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      out_rec = temp_rec;
    }
  }

  return hit_anything;
}

// PDF value is averaged over all sub-objects.
__device__ inline double
cuda_hittable_list_pdf_value(const void *ptr, const CudaPoint3 &origin,
                             const CudaVec3 &direction) {
  const auto *list = static_cast<const CudaHittableListData *>(ptr);
  if (list->count == 0)
    return 0.0;

  double weight = 1.0 / list->count;
  double sum = 0.0;

  for (int i = 0; i < list->count; i++) {
    sum +=
        weight * cuda_hittable_pdf_value(list->hittables[i], origin, direction);
  }

  return sum;
}

// Random direction toward one random object in the list.
__device__ inline CudaVec3 cuda_hittable_list_random(const void *ptr,
                                                     const CudaPoint3 &origin) {
  const auto *list = static_cast<const CudaHittableListData *>(ptr);
  int i = curand_uniform_double(&thread_rng_state) * list->count;
  i = clamp(i, 0, list->count - 1);
  return cuda_hittable_random(list->hittables[i], origin);
}

// Bounding box of the whole list.
__device__ inline CudaAABB cuda_hittable_list_bounding_box(const void *ptr) {
  return cuda_hittable_list_bbox((CudaHittableListData *)ptr);
}

// Static global vtable for all CudaHittableList instances.
__device__ __constant__ CudaHittableVTable cuda_hittable_list_vtable = {
    cuda_hittable_list_hit, cuda_hittable_list_pdf_value,
    cuda_hittable_list_random, cuda_hittable_list_bounding_box};

// Host-side construction function.
inline void create_cuda_hittable_list(CudaHittable *out_wrapper,
                                      CudaHittableListData *out_data,
                                      CudaHittable *device_hittables,
                                      int num_hittables) {
  assert(num_hittables <= MAX_HITTABLES_PER_LIST);
  cudaMemcpy(&out_data->hittables[0], device_hittables,
             sizeof(CudaHittable) * num_hittables, cudaMemcpyDeviceToDevice);

  out_data->count = num_hittables;

  // Optional: set bbox to something valid (can be updated later)
  out_data->bbox = CudaAABB();

  out_wrapper->data = out_data;
  cudaMemcpyFromSymbol(&out_wrapper->vtable, cuda_hittable_list_vtable,
                       sizeof(CudaHittableVTable));
}

#endif // USE_CUDA
