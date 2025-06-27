#pragma once

#ifdef USE_CUDA

#include "../core/RayConversions.cuh"
#include "../optimization/AABBConversions.cuh"
#include "../optimization/BVHNode.cuh"
#include "../optimization/BVHNode.hpp"
#include "../scene/materials/MaterialConverter.cuh"
#include "../scene/mediums/ConstantMedium.cuh"
#include "../scene/mediums/ConstantMedium.hpp"
#include "../scene/objects/Plane.cuh"
#include "../scene/objects/Plane.hpp"
#include "../scene/objects/RotateY.cuh"
#include "../scene/objects/RotateY.hpp"
#include "../scene/objects/Sphere.cuh"
#include "../scene/objects/Sphere.hpp"
#include "../scene/objects/Translate.cuh"
#include "../scene/objects/Translate.hpp"
#include "../scene/textures/TextureConverter.cuh"
#include "../utils/math/Utility.cuh"
#include "../utils/math/Vec3Conversions.cuh"
#include "../utils/memory/CudaMemoryUtility.cuh"
#include "../utils/memory/CudaSceneContext.cuh"
#include "Hittable.cuh"
#include "HittableList.hpp"
#include "HittableTypes.hpp"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <typeinfo>

// Forward declarations.
class Hittable;

// Hittable object conversion system.
class HittableConverter {
private:
  MaterialConverter &m_material_converter;
  TextureConverter &m_texture_converter;
  CudaSceneContext &m_context;

public:
  HittableConverter(MaterialConverter &mat_converter,
                    TextureConverter &tex_converter, CudaSceneContext &ctx)
      : m_material_converter(mat_converter), m_texture_converter(tex_converter),
        m_context(ctx) {}

  // Convert CPU Hittable to CUDA Hittable with type detection.
  CudaHittable cpu_to_cuda_hittable(HittablePtr cpu_hittable) {
    CudaHittable cuda_hittable;

    if (auto sphere = dynamic_cast<const Sphere *>(cpu_hittable.get())) {
      cuda_hittable.type = CudaHittableType::HITTABLE_SPHERE;
      cuda_hittable.sphere = m_context.suballocate<CudaSphere>(1);
      CudaSphere host_sphere = cpu_to_cuda_sphere(*sphere);
      cudaMemcpyHostToDeviceSafe(cuda_hittable.sphere, &host_sphere, 1);
    } else if (auto plane = dynamic_cast<const Plane *>(cpu_hittable.get())) {
      cuda_hittable.type = CudaHittableType::HITTABLE_PLANE;
      cuda_hittable.plane = m_context.suballocate<CudaPlane>(1);
      CudaPlane host_plane = cpu_to_cuda_plane(*plane);
      cudaMemcpyHostToDeviceSafe(cuda_hittable.plane, &host_plane, 1);
    } else if (auto bvh_node =
                   dynamic_cast<const BVHNode *>(cpu_hittable.get())) {
      cuda_hittable.type = CudaHittableType::HITTABLE_BVH_NODE;
      cuda_hittable.bvh_node = m_context.suballocate<CudaBVHNode>(1);
      CudaBVHNode host_bvh_node = cpu_to_cuda_bvh_node(*bvh_node);
      cudaMemcpyHostToDeviceSafe(cuda_hittable.bvh_node, &host_bvh_node, 1);
    } else if (auto constant_medium =
                   dynamic_cast<const ConstantMedium *>(cpu_hittable.get())) {
      cuda_hittable.type = CudaHittableType::HITTABLE_CONSTANT_MEDIUM;
      cuda_hittable.constant_medium =
          m_context.suballocate<CudaConstantMedium>(1);
      CudaConstantMedium host_constant_medium =
          cpu_to_cuda_constant_medium(*constant_medium);
      cudaMemcpyHostToDeviceSafe(cuda_hittable.constant_medium,
                                 &host_constant_medium, 1);
    } else if (auto translate =
                   dynamic_cast<const Translate *>(cpu_hittable.get())) {
      cuda_hittable.type = CudaHittableType::HITTABLE_TRANSLATE;
      cuda_hittable.translate = m_context.suballocate<CudaTranslate>(1);
      CudaTranslate host_translate = cpu_to_cuda_translate(*translate);
      cudaMemcpyHostToDeviceSafe(cuda_hittable.translate, &host_translate, 1);
    } else if (auto rotate_y =
                   dynamic_cast<const RotateY *>(cpu_hittable.get())) {
      cuda_hittable.type = CudaHittableType::HITTABLE_ROTATE_Y;
      cuda_hittable.rotate_y = m_context.suballocate<CudaRotateY>(1);
      CudaRotateY host_rotate_y = cpu_to_cuda_rotate_y(*rotate_y);
      cudaMemcpyHostToDeviceSafe(cuda_hittable.rotate_y, &host_rotate_y, 1);
    } else if (auto hittable_list =
                   dynamic_cast<const HittableList *>(cpu_hittable.get())) {
      cuda_hittable.type = CudaHittableType::HITTABLE_LIST;
      const std::vector<HittablePtr> &cpu_objects =
          hittable_list->get_objects();
      int object_count = (int)cpu_objects.size();
      CudaHittable *cuda_hittables_buffer =
          m_context.suballocate<CudaHittable>(object_count);
      cuda_hittable.hittable_list = m_context.suballocate<CudaHittableList>(1);
      CudaHittableList host_hittable_list = cpu_to_cuda_hittable_list(
          *hittable_list, cuda_hittables_buffer, object_count);
      cudaMemcpyHostToDeviceSafe(cuda_hittable.hittable_list,
                                 &host_hittable_list, 1);
    } else {
      throw std::runtime_error(
          "HittableConversions.cuh::cpu_to_cuda_hittable: Unknown "
          "hittable type encountered during CPU to CUDA conversion. "
          "Unable to convert unrecognized hittable object.");
    }

    return cuda_hittable;
  }

private:
  // Convert CPU Sphere to CUDA Sphere POD struct.
  inline CudaSphere cpu_to_cuda_sphere(const Sphere &cpu_sphere) {
    Ray cpu_center = cpu_sphere.get_center();
    double radius = cpu_sphere.get_radius();
    AABB bbox = cpu_sphere.get_bounding_box();
    MaterialPtr material = cpu_sphere.get_material();

    CudaRay cuda_center = cpu_to_cuda_ray(cpu_center);
    CudaAABB cuda_bbox = cpu_to_cuda_aabb(bbox);
    size_t material_index = m_material_converter.get_material_index(material);

    return cuda_make_sphere(cuda_center, radius, material_index, cuda_bbox);
  }

  // Convert CPU Plane to CUDA Plane POD struct.
  inline CudaPlane cpu_to_cuda_plane(const Plane &cpu_plane) {
    Point3 corner = cpu_plane.get_corner();
    Vec3 u_side = cpu_plane.get_u_side();
    Vec3 v_side = cpu_plane.get_v_side();
    MaterialPtr material = cpu_plane.get_material();
    AABB cpu_bbox = cpu_plane.get_bounding_box();

    CudaPoint3 cuda_corner = cpu_to_cuda_vec3(corner);
    CudaVec3 cuda_u_side = cpu_to_cuda_vec3(u_side);
    CudaVec3 cuda_v_side = cpu_to_cuda_vec3(v_side);
    CudaAABB cuda_bbox = cpu_to_cuda_aabb(cpu_bbox);

    size_t material_index = m_material_converter.get_material_index(material);
    return cuda_make_plane(cuda_corner, cuda_u_side, cuda_v_side,
                           material_index, cuda_bbox);
  }

  // Convert CPU RotateY to CUDA RotateY.
  inline CudaRotateY cpu_to_cuda_rotate_y(const RotateY &cpu_rotate_y) {
    // Extract properties from CPU RotateY using getters.
    HittablePtr cpu_object = cpu_rotate_y.get_object();
    double angle_degrees = cpu_rotate_y.get_angle();
    AABB cpu_bbox = cpu_rotate_y.get_bounding_box();

    CudaHittable *cuda_object = m_context.suballocate<CudaHittable>(1);
    CudaHittable host_object = cpu_to_cuda_hittable(cpu_object);
    cudaMemcpyHostToDeviceSafe(cuda_object, &host_object, 1);

    double radians = angle_degrees * CUDA_PI / 180.0;
    CudaAABB cuda_bbox = cpu_to_cuda_aabb(cpu_bbox);

    return cuda_make_rotate_y(cuda_object, sin(radians), cos(radians),
                              cuda_bbox);
  }

  // Convert CPU Translate to CUDA Translate POD struct.
  inline CudaTranslate cpu_to_cuda_translate(const Translate &cpu_translate) {
    HittablePtr cpu_object = cpu_translate.get_object();
    Vec3 cpu_offset = cpu_translate.get_offset();

    CudaHittable *cuda_object = m_context.suballocate<CudaHittable>(1);
    CudaHittable host_object = cpu_to_cuda_hittable(cpu_object);
    cudaMemcpyHostToDeviceSafe(cuda_object, &host_object, 1);

    CudaVec3 cuda_offset = cpu_to_cuda_vec3(cpu_offset);
    CudaAABB cuda_bbox = cpu_to_cuda_aabb(cpu_translate.get_bounding_box());

    return cuda_make_translate(cuda_object, cuda_offset, cuda_bbox);
  }

  // Convert CPU ConstantMedium to CUDA ConstantMedium.
  inline CudaConstantMedium
  cpu_to_cuda_constant_medium(const ConstantMedium &cpu_constant_medium) {
    HittablePtr cpu_boundary = cpu_constant_medium.get_boundary();
    double density = cpu_constant_medium.get_density();
    MaterialPtr cpu_phase_function = cpu_constant_medium.get_phase_function();

    CudaHittable *cuda_boundary = m_context.suballocate<CudaHittable>(1);
    CudaHittable host_boundary = cpu_to_cuda_hittable(cpu_boundary);
    cudaMemcpyHostToDeviceSafe(cuda_boundary, &host_boundary, 1);

    size_t material_index =
        m_material_converter.get_material_index(cpu_phase_function);
    return cuda_make_constant_medium(cuda_boundary, density, material_index);
  }

  // Convert CPU BVHNode to CUDA BVHNode POD struct.
  inline CudaBVHNode cpu_to_cuda_bvh_node(const BVHNode &cpu_bvh_node) {
    CudaHittable *left = m_context.suballocate<CudaHittable>(1);
    CudaHittable *right = m_context.suballocate<CudaHittable>(1);
    CudaHittable host_left = cpu_to_cuda_hittable(cpu_bvh_node.get_left());
    CudaHittable host_right = cpu_to_cuda_hittable(cpu_bvh_node.get_right());
    cudaMemcpyHostToDeviceSafe(left, &host_left, 1);
    cudaMemcpyHostToDeviceSafe(right, &host_right, 1);

    bool is_leaf = host_left.type != CudaHittableType::HITTABLE_BVH_NODE &&
                   host_right.type != CudaHittableType::HITTABLE_BVH_NODE;

    AABB cpu_bbox = cpu_bvh_node.get_bounding_box();
    CudaAABB cuda_bbox = cpu_to_cuda_aabb(cpu_bbox);

    return cuda_make_bvh_node(left, right, is_leaf, cuda_bbox);
  }

  // Convert CPU HittableList to CUDA HittableList.
  // NOTE: This function assumes that the cuda_hittables_buffer has been
  // dynamically allocated BEFORE calling, so there's no allocations within this
  // function.
  inline CudaHittableList
  cpu_to_cuda_hittable_list(const HittableList &cpu_list,
                            CudaHittable *cuda_hittables_buffer,
                            int num_objects) {
    const std::vector<HittablePtr> &cpu_objects = cpu_list.get_objects();
    AABB cpu_bbox = cpu_list.get_bounding_box();

    // Convert each hittable object on host first, then copy to device.
    std::vector<CudaHittable> host_hittables(num_objects);
    for (int i = 0; i < num_objects; i++) {
      const HittablePtr &cpu_object = cpu_objects[i];
      host_hittables[i] = cpu_to_cuda_hittable(cpu_object);
    }
    CudaAABB cuda_bbox = cpu_to_cuda_aabb(cpu_bbox);

    // Copy all hittables to device buffer at once.
    cudaMemcpyHostToDeviceSafe(cuda_hittables_buffer, host_hittables.data(),
                               num_objects);

    // Create CUDA hittable list using POD initialization function.
    return cuda_make_hittable_list(cuda_hittables_buffer, num_objects,
                                   cuda_bbox);
  }
};

#endif // USE_CUDA