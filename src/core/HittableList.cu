#ifdef USE_CUDA

#include "Hittable.cuh"
#include "HittableList.cuh"

__device__ CudaHittableList::CudaHittableList(CudaHittable *_hittables,
                                              int _num_hittables) {
  assert(_num_hittables <= MAX_HITTABLES_PER_LIST);

  hittables = _hittables;
  count = _num_hittables;

  // Calculate bounding box from all objects.
  bbox = CudaAABB();
  for (int i = 0; i < count; i++) {
    bbox = CudaAABB(bbox, hittables[i].get_bounding_box());
  }
}

__device__ bool CudaHittableList::hit(const CudaRay &ray, CudaInterval t_range,
                                      CudaHitRecord &out_rec,
                                      curandState *rand_state) const {
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

__device__ double CudaHittableList::pdf_value(const CudaPoint3 &origin,
                                              const CudaVec3 &direction) const {
  // For a list of hittable objects, the pdf value biasing should be just the
  // average of all the PDF value biasing of the inner hittable objects.

  if (count == 0)
    return 0.0;

  double weight = 1.0 / count;
  double sum = 0.0;

  for (int i = 0; i < count; i++) {
    sum += weight * hittables[i].pdf_value(origin, direction);
  }

  return sum;
}

__device__ CudaVec3 CudaHittableList::random(const CudaPoint3 &origin,
                                             curandState *state) const {
  // Randomly chooses one object in the list and returns a direction vector
  // sampled from it.

  int i = (int)(cuda_random_double(state) * count);
  i = (i < 0) ? 0 : ((i >= count) ? count - 1 : i);
  return hittables[i].random(origin, state);
}

#endif // USE_CUDA