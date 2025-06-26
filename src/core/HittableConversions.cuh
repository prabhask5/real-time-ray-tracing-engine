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

  if (auto sphere = dynamic_cast<const Sphere *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_SPHERE;
    cuda_hittable.sphere = new CudaSphere();
    *cuda_hittable.sphere = cpu_to_cuda_sphere(*sphere);
  } else if (auto plane = dynamic_cast<const Plane *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_PLANE;
    cuda_hittable.plane = new CudaPlane();
    *cuda_hittable.plane = cpu_to_cuda_plane(*plane);
  } else if (auto bvh_node = dynamic_cast<const BVHNode *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_BVH_NODE;
    cuda_hittable.bvh_node = new CudaBVHNode();
    *cuda_hittable.bvh_node = cpu_to_cuda_bvh_node(*bvh_node);
  } else if (auto constant_medium =
                 dynamic_cast<const ConstantMedium *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_CONSTANT_MEDIUM;
    cuda_hittable.constant_medium = new CudaConstantMedium();
    *cuda_hittable.constant_medium =
        cpu_to_cuda_constant_medium(*constant_medium);
  } else if (auto translate = dynamic_cast<const Translate *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_TRANSLATE;
    cuda_hittable.translate = new CudaTranslate();
    *cuda_hittable.translate = cpu_to_cuda_translate(*translate);
  } else if (auto rotate_y = dynamic_cast<const RotateY *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_ROTATE_Y;
    cuda_hittable.rotate_y = new CudaRotateY();
    *cuda_hittable.rotate_y = cpu_to_cuda_rotate_y(*rotate_y);
  } else if (auto hittable_list =
                 dynamic_cast<const HittableList *>(&cpu_hittable)) {
    cuda_hittable.type = CudaHittableType::HITTABLE_LIST;
    CudaHittable nested_buffer[MAX_HITTABLES_PER_LIST];
    cuda_hittable.hittable_list = new CudaHittableList();
    *cuda_hittable.hittable_list =
        cpu_to_cuda_hittable_list(*hittable_list, nested_buffer);
  } else {
    throw std::runtime_error(
        "HittableConversions.cuh::cpu_to_cuda_hittable: Unknown hittable type "
        "encountered during CPU to CUDA conversion. Unable to convert "
        "unrecognized hittable object.");
  }

  return cuda_hittable;
}

#endif // USE_CUDA