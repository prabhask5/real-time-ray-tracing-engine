#pragma once

#ifdef USE_CUDA

#include "Hittable.cuh"
#include "HittableList.hpp"
#include "HittableTypes.hpp"

// Forward declarations for conversion functions - defined in
// HittableConversions.cuh.
CudaHittable cpu_to_cuda_hittable(const Hittable &cpu_hittable);

// Convert CPU HittableList to CUDA HittableList.
inline CudaHittableList
cpu_to_cuda_hittable_list(const HittableList &cpu_list,
                          CudaHittable *cuda_hittables_buffer,
                          int max_objects = MAX_HITTABLES_PER_LIST) {
  const std::vector<HittablePtr> &cpu_objects = cpu_list.get_objects();
  int object_count = std::min((int)cpu_objects.size(), max_objects);

  // Convert each hittable object.
  for (int i = 0; i < object_count; i++) {
    const HittablePtr &cpu_object = cpu_objects[i];

    // Convert each object type properly with its actual material.
    cuda_hittables_buffer[i] = cpu_to_cuda_hittable(*cpu_object);
  }

  // Create CUDA hittable list using POD initialization function.
  return cuda_make_hittable_list(cuda_hittables_buffer, object_count);
}

#endif // USE_CUDA