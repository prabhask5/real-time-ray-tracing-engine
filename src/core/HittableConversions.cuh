#pragma once

#ifdef USE_CUDA

#include <stdexcept>

#include "../optimization/BVHNode.hpp"
#include "../optimization/BVHNodeConversions.cuh"
#include "../scene/materials/LambertianMaterial.hpp"
#include "../scene/mediums/ConstantMediumConversions.cuh"
#include "../scene/objects/Plane.hpp"
#include "../scene/objects/PlaneConversions.cuh"
#include "../scene/objects/RotateYConversions.cuh"
#include "../scene/objects/Sphere.hpp"
#include "../scene/objects/SphereConversions.cuh"
#include "../scene/objects/TranslateConversions.cuh"
#include "../scene/textures/SolidColorTexture.hpp"
#include "Hittable.cuh"
#include "HittableList.hpp"
#include "HittableListConversions.cuh"
#include "HittableTypes.hpp"

// Forward declarations.
class Hittable;

// Convert CPU Hittable to CUDA Hittable with type detection.
inline CudaHittable cpu_to_cuda_hittable(const Hittable &cpu_hittable) {
  CudaHittable cuda_hittable;

  // Use runtime type identification to determine object type.
  if (const Sphere *sphere = dynamic_cast<const Sphere *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_SPHERE;
    cuda_hittable.sphere = new CudaSphere(cpu_to_cuda_sphere(*sphere));
  } else if (const Plane *plane = dynamic_cast<const Plane *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_PLANE;
    cuda_hittable.plane = new CudaPlane(cpu_to_cuda_plane(*plane));
  } else if (const BVHNode *bvh_node =
                 dynamic_cast<const BVHNode *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_BVH_NODE;
    cuda_hittable.bvh_node = new CudaBVHNode(cpu_to_cuda_bvh_node(*bvh_node));
  } else if (const ConstantMedium *constant_medium =
                 dynamic_cast<const ConstantMedium *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_CONSTANT_MEDIUM;
    cuda_hittable.constant_medium =
        new CudaConstantMedium(cpu_to_cuda_constant_medium(*constant_medium));
  } else if (const Translate *translate =
                 dynamic_cast<const Translate *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_TRANSLATE;
    cuda_hittable.translate =
        new CudaTranslate(cpu_to_cuda_translate(*translate));
  } else if (const RotateY *rotate_y =
                 dynamic_cast<const RotateY *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_ROTATE_Y;
    cuda_hittable.rotate_y = new CudaRotateY(cpu_to_cuda_rotate_y(*rotate_y));
  } else if (const HittableList *hittable_list =
                 dynamic_cast<const HittableList *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_LIST;
    CudaHittable nested_buffer[MAX_HITTABLES_PER_LIST];
    cuda_hittable.hittable_list = new CudaHittableList(
        cpu_to_cuda_hittable_list(*hittable_list, nested_buffer));
  } else {
    throw std::runtime_error(
        "HittableConversions.cuh::cpu_to_cuda_hittable: Unknown hittable type "
        "encountered during CPU to CUDA conversion. Unable to convert "
        "unrecognized hittable object.");
  }

  return cuda_hittable;
}

// Convert CUDA Hittable to CPU Hittable.
inline HittablePtr cuda_to_cpu_hittable(const CudaHittable &cuda_hittable) {
  switch (cuda_hittable.type) {
  case CudaHittableType::HITTABLE_SPHERE:
    return cuda_to_cpu_sphere(*cuda_hittable.sphere);
  case CudaHittableType::HITTABLE_PLANE:
    return cuda_to_cpu_plane(*cuda_hittable.plane);
  case CudaHittableType::HITTABLE_BVH_NODE:
    return std::make_shared<BVHNode>(
        cuda_to_cpu_bvh_node(*cuda_hittable.bvh_node));
  case CudaHittableType::HITTABLE_CONSTANT_MEDIUM:
    return std::make_shared<ConstantMedium>(
        cuda_to_cpu_constant_medium(*cuda_hittable.constant_medium));
  case CudaHittableType::HITTABLE_TRANSLATE:
    return std::make_shared<Translate>(
        cuda_to_cpu_translate(*cuda_hittable.translate));
  case CudaHittableType::HITTABLE_ROTATE_Y:
    return std::make_shared<RotateY>(
        cuda_to_cpu_rotate_y(*cuda_hittable.rotate_y));
  case CudaHittableType::HITTABLE_LIST:
    return std::make_shared<HittableList>(
        cuda_to_cpu_hittable_list(*cuda_hittable.hittable_list));
  default:
    throw std::runtime_error(
        "HittableConversions.cuh::cuda_to_cpu_hittable: Unknown CUDA hittable "
        "type encountered during CUDA to CPU conversion. Invalid hittable type "
        "in switch statement.");
  }
}

#endif // USE_CUDA