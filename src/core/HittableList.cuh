#pragma once

#ifdef USE_CUDA

// Maximum objects per list.
static const int MAX_HITTABLES_PER_LIST = 1024;

// CUDA-compatible struct representing a list of hittables.
// Method implementations are defined after CudaHittable definition.
struct CudaHittableList {
  CudaHittable *hittables; // Pointer to array of hittables
  int count;
  CudaAABB bbox;

  __host__ __device__ CudaHittableList() {} // Default constructor.
  __device__ CudaHittableList(CudaHittable *_hittables, int _num_hittables);

  __device__ bool hit(const CudaRay &ray, CudaInterval t_range,
                      CudaHitRecord &out_rec, curandState *rand_state) const;

  __device__ double pdf_value(const CudaPoint3 &origin,
                              const CudaVec3 &direction) const;

  __device__ CudaVec3 random(const CudaPoint3 &origin,
                             curandState *state) const;

  __device__ inline CudaAABB get_bounding_box() const { return bbox; }
};

#endif // USE_CUDA