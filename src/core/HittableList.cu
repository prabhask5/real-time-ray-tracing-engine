#ifdef USE_CUDA

#include "../utils/math/Utility.cuh"
#include "../utils/memory/CudaMemoryUtility.cuh"
#include "Hittable.cuh"
#include "HittableList.cuh"
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <sstream>

__device__ bool cuda_hittable_list_hit(const CudaHittableList &list,
                                       const CudaRay &ray, CudaInterval t_range,
                                       CudaHitRecord &out_rec,
                                       curandState *rand_state) {
  CudaHitRecord temp_rec;
  bool hit_anything = false;
  double closest_so_far = t_range.max;

  for (int i = 0; i < list.count; i++) {
    if (cuda_hittable_hit(list.hittables[i], ray,
                          cuda_make_interval(t_range.min, closest_so_far),
                          temp_rec, rand_state)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      out_rec = temp_rec;
    }
  }

  return hit_anything;
}

__device__ double cuda_hittable_list_pdf_value(const CudaHittableList &list,
                                               const CudaPoint3 &origin,
                                               const CudaVec3 &direction) {
  // For a list of hittable objects, the pdf value biasing should be just the
  // average of all the PDF value biasing of the inner hittable objects.

  if (list.count == 0)
    return 0.0;

  double weight = 1.0 / list.count;
  double sum = 0.0;

  for (int i = 0; i < list.count; i++) {
    sum +=
        weight * cuda_hittable_pdf_value(list.hittables[i], origin, direction);
  }

  return sum;
}

__device__ CudaVec3 cuda_hittable_list_random(const CudaHittableList &list,
                                              const CudaPoint3 &origin,
                                              curandState *state) {
  // Randomly chooses one object in the list and returns a direction vector
  // sampled from it.

  int i = (int)(cuda_random_double(state) * list.count);
  i = (i < 0) ? 0 : ((i >= list.count) ? list.count - 1 : i);
  return cuda_hittable_random(list.hittables[i], origin, state);
}

// JSON serialization function for CudaHittableList.
std::string cuda_json_hittable_list(const CudaHittableList &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaHittableList\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"hittables\":[";
  if (obj.hittables && obj.count > 0) {
    CudaHittable *host_hittables = new CudaHittable[obj.count];
    cudaMemcpyDeviceToHostSafe(host_hittables, obj.hittables, obj.count);
    for (int i = 0; i < obj.count; ++i) {
      if (i > 0)
        oss << ",";
      oss << cuda_json_hittable(host_hittables[i]);
    }
    delete[] host_hittables;
  }
  oss << "],";
  oss << "\"count\":" << obj.count << ",";
  oss << "\"bbox\":" << cuda_json_aabb(obj.bbox);
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA