#pragma once

#ifdef USE_CUDA

#include "../optimization/AABBConversions.cuh"
#include "../optimization/BVHNodeConversions.cuh"
#include "../scene/mediums/ConstantMediumConversions.cuh"
#include "../scene/objects/PlaneConversions.cuh"
#include "../scene/objects/RotateYConversions.cuh"
#include "../scene/objects/SphereConversions.cuh"
#include "../scene/objects/TranslateConversions.cuh"
#include "Hittable.cuh"
#include "HittableList.cuh"
#include "HittableList.hpp"

// Maximum number of hittables that can be converted in a single operation.
constexpr int MAX_HITTABLE_CONVERSION_COUNT = 64;

// Convert CPU HittableList to CUDA HittableList.
inline CudaHittableList
cpu_to_cuda_hittable_list(const HittableList &cpu_list,
                          CudaHittable *cuda_hittables_buffer,
                          int max_objects = MAX_HITTABLES_PER_LIST) {
  const auto &cpu_objects = cpu_list.get_objects();
  int object_count = std::min((int)cpu_objects.size(), max_objects);

  // Convert each hittable object.
  for (int i = 0; i < object_count; i++) {
    const auto &cpu_object = cpu_objects[i];

    // Convert each object type properly with its actual material.
    if (auto sphere = std::dynamic_pointer_cast<Sphere>(cpu_object)) {
      cuda_hittables_buffer[i].type = CudaHittableType::HITTABLE_SPHERE;
      cuda_hittables_buffer[i].sphere = cpu_to_cuda_sphere(*sphere);
    } else if (auto plane = std::dynamic_pointer_cast<Plane>(cpu_object)) {
      cuda_hittables_buffer[i].type = CudaHittableType::HITTABLE_PLANE;
      cuda_hittables_buffer[i].plane = cpu_to_cuda_plane(*plane);
    } else if (auto bvh_node = std::dynamic_pointer_cast<BVHNode>(cpu_object)) {
      cuda_hittables_buffer[i].type = CudaHittableType::HITTABLE_BVH_NODE;
      cuda_hittables_buffer[i].bvh_node = cpu_to_cuda_bvh_node(*bvh_node);
    } else if (auto constant_medium =
                   std::dynamic_pointer_cast<ConstantMedium>(cpu_object)) {
      cuda_hittables_buffer[i].type =
          CudaHittableType::HITTABLE_CONSTANT_MEDIUM;
      cuda_hittables_buffer[i].constant_medium =
          cpu_to_cuda_constant_medium(*constant_medium);
    } else if (auto translate =
                   std::dynamic_pointer_cast<Translate>(cpu_object)) {
      cuda_hittables_buffer[i].type = CudaHittableType::HITTABLE_TRANSLATE;
      cuda_hittables_buffer[i].translate = cpu_to_cuda_translate(*translate);
    } else if (auto rotate_y = std::dynamic_pointer_cast<RotateY>(cpu_object)) {
      cuda_hittables_buffer[i].type = CudaHittableType::HITTABLE_ROTATE_Y;
      cuda_hittables_buffer[i].rotate_y = cpu_to_cuda_rotate_y(*rotate_y);
    } else if (auto hittable_list =
                   std::dynamic_pointer_cast<HittableList>(cpu_object)) {
      cuda_hittables_buffer[i].type = CudaHittableType::HITTABLE_LIST;
      CudaHittable nested_buffer[MAX_HITTABLES_PER_LIST];
      cuda_hittables_buffer[i].hittable_list =
          cpu_to_cuda_hittable_list(*hittable_list, nested_buffer);
    } else {
      // Default fallback - create minimal sphere with appropriate material.
      CudaMaterial default_material =
          cuda_make_lambertian_material(Color(0.5, 0.5, 0.5));
      cuda_hittables_buffer[i].type = CudaHittableType::HITTABLE_SPHERE;
      cuda_hittables_buffer[i].sphere =
          create_cuda_sphere_static(Point3(0, 0, 0), 0.001, default_material);
    }
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

    switch (cuda_obj.type) {
    case CudaHittableType::HITTABLE_SPHERE: {
      auto cpu_sphere = cuda_to_cpu_sphere(cuda_obj.sphere);
      cpu_list.add(cpu_sphere);
      break;
    }
    case CudaHittableType::HITTABLE_PLANE: {
      auto cpu_plane = cuda_to_cpu_plane(cuda_obj.plane);
      cpu_list.add(cpu_plane);
      break;
    }
    case CudaHittableType::HITTABLE_BVH_NODE: {
      auto cpu_bvh_node = cuda_to_cpu_bvh_node(cuda_obj.bvh_node);
      cpu_list.add(cpu_bvh_node);
      break;
    }
    case CudaHittableType::HITTABLE_CONSTANT_MEDIUM: {
      auto cpu_constant_medium =
          cuda_to_cpu_constant_medium(cuda_obj.constant_medium);
      cpu_list.add(cpu_constant_medium);
      break;
    }
    case CudaHittableType::HITTABLE_ROTATE_Y: {
      auto cpu_rotate_y = cuda_to_cpu_rotate_y(cuda_obj.rotate_y);
      cpu_list.add(cpu_rotate_y);
      break;
    }
    case CudaHittableType::HITTABLE_TRANSLATE: {
      auto cpu_translate = cuda_to_cpu_translate(cuda_obj.translate);
      cpu_list.add(cpu_translate);
      break;
    }
    case CudaHittableType::HITTABLE_LIST: {
      auto cpu_hittable_list =
          cuda_to_cpu_hittable_list(cuda_obj.hittable_list);
      cpu_list.add(cpu_hittable_list);
      break;
    }
    default:
      // Create default sphere for unknown types.
      auto default_material =
          std::make_shared<LambertianMaterial>(Color(0.5, 0.5, 0.5));
      auto default_sphere =
          std::make_shared<Sphere>(Point3(0, 0, 0), 1.0, default_material);
      cpu_list.add(default_sphere);
      break;
    }
  }

  return cpu_list;
}

// Create empty CUDA hittable list.
__host__ __device__ inline CudaHittableList create_empty_cuda_hittable_list() {
  CudaHittableList cuda_list;
  cuda_list.count = 0;
  cuda_list.bbox = CudaAABB();
  return cuda_list;
}

// Create hittable list with provided buffer.
__host__ __device__ inline CudaHittableList
create_cuda_hittable_list_with_buffer(CudaHittable *hittables_buffer,
                                      int object_count) {
  CudaHittableList cuda_list;
  cuda_list.count = std::min(object_count, MAX_HITTABLES_PER_LIST);

  // Copy objects from buffer.
  for (int i = 0; i < cuda_list.count; i++) {
    cuda_list.hittables[i] = hittables_buffer[i];
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

// Batch conversion functions.
void batch_cpu_to_cuda_hittable_list(const HittableList *cpu_lists,
                                     CudaHittableList *cuda_lists, int count);
void batch_cuda_to_cpu_hittable_list(const CudaHittableList *cuda_lists,
                                     HittableList *cpu_lists, int count);

#endif // USE_CUDA