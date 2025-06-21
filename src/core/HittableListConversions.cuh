#pragma once

#ifdef USE_CUDA

#include "Hittable.cuh"
#include "HittableList.hpp"
#include "HittableTypes.hpp"

// Forward declarations for conversion functions - defined in
// HittableConversions.cuh.
CudaHittable cpu_to_cuda_hittable(const Hittable &cpu_hittable);
HittablePtr cuda_to_cpu_hittable(const CudaHittable &cuda_hittable);

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

  // Create CUDA hittable list.
  CudaHittableList cuda_list;
  cuda_list.count = object_count;

  // Copy objects to the list.
  for (int i = 0; i < object_count && i < MAX_HITTABLES_PER_LIST; i++) {
    cuda_list.hittables[i] = cuda_hittables_buffer[i];
  }

  // Calculate bounding box.
  cuda_list.bbox = CudaAABB();
  for (int i = 0; i < cuda_list.count; i++) {
    CudaAABB obj_bbox = cuda_list.hittables[i].get_bounding_box();
    if (i == 0) {
      cuda_list.bbox = obj_bbox;
    } else {
      cuda_list.bbox = CudaAABB(cuda_list.bbox, obj_bbox);
    }
  }

  return cuda_list;
}

// Convert CUDA HittableList to CPU HittableList.
inline HittableList
cuda_to_cpu_hittable_list(const CudaHittableList &cuda_list) {
  HittableList cpu_list;

  for (int i = 0; i < cuda_list.count; i++) {
    const CudaHittable &cuda_obj = cuda_list.hittables[i];

    HittablePtr cpu_obj = cuda_to_cpu_hittable(cuda_obj);
    cpu_list.add(cpu_obj);
  }

  return cpu_list;
}

// Create empty CUDA hittable list.
__device__ inline CudaHittableList create_empty_cuda_hittable_list() {
  CudaHittableList cuda_list;
  cuda_list.count = 0;
  cuda_list.bbox = CudaAABB();
  return cuda_list;
}

#endif // USE_CUDA