#pragma once

#ifdef USE_CUDA

#include "../optimization/BVHNodeConversions.cuh"
#include "../scene/mediums/ConstantMediumConversions.cuh"
#include "../scene/objects/PlaneConversions.cuh"
#include "../scene/objects/RotateYConversions.cuh"
#include "../scene/objects/SphereConversions.cuh"
#include "../scene/objects/TranslateConversions.cuh"
#include "Hittable.cuh"
#include "HittableListConversions.cuh"
#include "HittableTypes.hpp"

// Forward declarations.
class Hittable;

// Convert CPU Hittable to CUDA Hittable with type detection.
inline CudaHittable cpu_to_cuda_hittable(const HittablePtr &cpu_hittable) {
  CudaHittable cuda_hittable;

  if (!cpu_hittable) {
    // Create default sphere.
    cuda_hittable.type = CudaHittableType::HITTABLE_SPHERE;
    cuda_hittable.sphere = create_cuda_sphere_static(
        CudaPoint3(0, 0, 0), 1.0,
        cuda_make_lambertian_material(CudaColor(0.5, 0.5, 0.5)));
    return cuda_hittable;
  }

  // Use runtime type identification to determine object type.
  if (auto sphere = std::dynamic_pointer_cast<Sphere>(cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_SPHERE;
    cuda_hittable.sphere = cpu_to_cuda_sphere(*sphere);
  } else if (auto plane = std::dynamic_pointer_cast<Plane>(cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_PLANE;
    cuda_hittable.plane = cpu_to_cuda_plane(*plane);
  } else if (auto bvh_node = std::dynamic_pointer_cast<BVHNode>(cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_BVH_NODE;
    cuda_hittable.bvh_node = cpu_to_cuda_bvh_node(*bvh_node);
  } else if (auto constant_medium =
                 std::dynamic_pointer_cast<ConstantMedium>(cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_CONSTANT_MEDIUM;
    cuda_hittable.constant_medium =
        cpu_to_cuda_constant_medium(*constant_medium);
  } else if (auto translate =
                 std::dynamic_pointer_cast<Translate>(cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_TRANSLATE;
    cuda_hittable.translate = cpu_to_cuda_translate(*translate);
  } else if (auto rotate_y = std::dynamic_pointer_cast<RotateY>(cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_ROTATE_Y;
    cuda_hittable.rotate_y = cpu_to_cuda_rotate_y(*rotate_y);
  } else if (auto hittable_list =
                 std::dynamic_pointer_cast<HittableList>(cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_LIST;
    cuda_hittable.hittable_list = cpu_to_cuda_hittable_list(*hittable_list);
  } else {
    // Default fallback to sphere.
    cuda_hittable.type = HITTABLE_SPHERE;
    cuda_hittable.sphere = create_cuda_sphere_static(
        CudaPoint3(0, 0, 0), 1.0,
        cuda_make_lambertian_material(CudaColor(0.5, 0.5, 0.5)));
  }

  return cuda_hittable;
}

// Convert CUDA Hittable to CPU Hittable.
inline HittablePtr cuda_to_cpu_hittable(const CudaHittable &cuda_hittable) {
  switch (cuda_hittable.type) {
  case CudaHittableType::HITTABLE_SPHERE:
    return std::make_shared<Sphere>(cuda_to_cpu_sphere(cuda_hittable.sphere));
  case CudaHittableType::HITTABLE_PLANE:
    return std::make_shared<Plane>(cuda_to_cpu_plane(cuda_hittable.plane));
  case CudaHittableType::HITTABLE_BVH_NODE:
    return std::make_shared<BVHNode>(
        cuda_to_cpu_bvh_node(cuda_hittable.bvh_node));
  case CudaHittableType::HITTABLE_CONSTANT_MEDIUM:
    return std::make_shared<ConstantMedium>(
        cuda_to_cpu_constant_medium(cuda_hittable.constant_medium));
  case CudaHittableType::HITTABLE_TRANSLATE:
    return std::make_shared<Translate>(
        cuda_to_cpu_translate(cuda_hittable.translate));
  case CudaHittableType::HITTABLE_ROTATE_Y:
    return std::make_shared<RotateY>(
        cuda_to_cpu_rotate_y(cuda_hittable.rotate_y));
  case CudaHittableType::HITTABLE_LIST:
    return std::make_shared<HittableList>(
        cuda_to_cpu_hittable_list(cuda_hittable.hittable_list));
  default:
    // Return default sphere.
    return std::make_shared<Sphere>(
        Point3(0, 0, 0), 1.0,
        std::make_shared<LambertianMaterial>(
            std::make_shared<SolidColorTexture>(Color(0.5, 0.5, 0.5))));
  }
}

#endif // USE_CUDA