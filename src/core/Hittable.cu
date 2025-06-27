#ifdef USE_CUDA

#include "../optimization/BVHNode.cuh"
#include "../scene/mediums/ConstantMedium.cuh"
#include "../scene/objects/Plane.cuh"
#include "../scene/objects/RotateY.cuh"
#include "../scene/objects/Sphere.cuh"
#include "../scene/objects/Translate.cuh"
#include "../utils/memory/CudaMemoryUtility.cuh"
#include "../utils/memory/CudaSceneContext.cuh"
#include "Hittable.cuh"
#include "HittableList.cuh"
#include <iomanip>
#include <sstream>

// CudaHittable constructors and dispatch methods implementation.

__device__ bool cuda_hittable_hit(const CudaHittable &hittable,
                                  const CudaRay &ray, CudaInterval t_range,
                                  CudaHitRecord &rec, curandState *rand_state) {
  switch (hittable.type) {
  case CudaHittableType::HITTABLE_SPHERE:
    return cuda_sphere_hit(*hittable.sphere, ray, t_range, rec, rand_state);
  case CudaHittableType::HITTABLE_PLANE:
    return cuda_plane_hit(*hittable.plane, ray, t_range, rec, rand_state);
  case CudaHittableType::HITTABLE_BVH_NODE:
    return cuda_bvh_node_hit(*hittable.bvh_node, ray, t_range, rec, rand_state);
  case CudaHittableType::HITTABLE_CONSTANT_MEDIUM:
    return cuda_constant_medium_hit(*hittable.constant_medium, ray, t_range,
                                    rec, rand_state);
  case CudaHittableType::HITTABLE_ROTATE_Y:
    return cuda_rotate_y_hit(*hittable.rotate_y, ray, t_range, rec, rand_state);
  case CudaHittableType::HITTABLE_TRANSLATE:
    return cuda_translate_hit(*hittable.translate, ray, t_range, rec,
                              rand_state);
  case CudaHittableType::HITTABLE_LIST:
    return cuda_hittable_list_hit(*hittable.hittable_list, ray, t_range, rec,
                                  rand_state);
  default:
    return false;
  }
}

__device__ double cuda_hittable_pdf_value(const CudaHittable &hittable,
                                          const CudaPoint3 &origin,
                                          const CudaVec3 &direction) {
  switch (hittable.type) {
  case CudaHittableType::HITTABLE_SPHERE:
    return cuda_sphere_pdf_value(*hittable.sphere, origin, direction);
  case CudaHittableType::HITTABLE_PLANE:
    return cuda_plane_pdf_value(*hittable.plane, origin, direction);
  case CudaHittableType::HITTABLE_BVH_NODE:
    return cuda_bvh_node_pdf_value(*hittable.bvh_node, origin, direction);
  case CudaHittableType::HITTABLE_ROTATE_Y:
    return cuda_rotate_y_pdf_value(*hittable.rotate_y, origin, direction);
  case CudaHittableType::HITTABLE_TRANSLATE:
    return cuda_translate_pdf_value(*hittable.translate, origin, direction);
  case CudaHittableType::HITTABLE_LIST:
    return cuda_hittable_list_pdf_value(*hittable.hittable_list, origin,
                                        direction);
  default:
    return 0.0;
  }
}

__device__ CudaVec3 cuda_hittable_random(const CudaHittable &hittable,
                                         const CudaPoint3 &origin,
                                         curandState *state) {
  switch (hittable.type) {
  case CudaHittableType::HITTABLE_SPHERE:
    return cuda_sphere_random(*hittable.sphere, origin, state);
  case CudaHittableType::HITTABLE_PLANE:
    return cuda_plane_random(*hittable.plane, origin, state);
  case CudaHittableType::HITTABLE_BVH_NODE:
    return cuda_bvh_node_random(*hittable.bvh_node, origin, state);
  case CudaHittableType::HITTABLE_ROTATE_Y:
    return cuda_rotate_y_random(*hittable.rotate_y, origin, state);
  case CudaHittableType::HITTABLE_TRANSLATE:
    return cuda_translate_random(*hittable.translate, origin, state);
  case CudaHittableType::HITTABLE_LIST:
    return cuda_hittable_list_random(*hittable.hittable_list, origin, state);
  default:
    return cuda_make_vec3(1.0, 0.0, 0.0);
  }
}

__host__ __device__ CudaAABB
cuda_hittable_get_bounding_box(const CudaHittable &hittable) {
  switch (hittable.type) {
  case CudaHittableType::HITTABLE_SPHERE:
    return cuda_sphere_get_bounding_box(*hittable.sphere);
  case CudaHittableType::HITTABLE_PLANE:
    return cuda_plane_get_bounding_box(*hittable.plane);
  case CudaHittableType::HITTABLE_BVH_NODE:
    return cuda_bvh_node_get_bounding_box(*hittable.bvh_node);
  case CudaHittableType::HITTABLE_CONSTANT_MEDIUM:
    return cuda_constant_medium_get_bounding_box(*hittable.constant_medium);
  case CudaHittableType::HITTABLE_ROTATE_Y:
    return cuda_rotate_y_get_bounding_box(*hittable.rotate_y);
  case CudaHittableType::HITTABLE_TRANSLATE:
    return cuda_translate_get_bounding_box(*hittable.translate);
  case CudaHittableType::HITTABLE_LIST:
    return cuda_hittable_list_get_bounding_box(*hittable.hittable_list);
  default:
    return cuda_make_aabb();
  }
}

// JSON serialization function for CudaHittable.
std::string cuda_json_hittable(const CudaHittable &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaHittable\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"hittable_type\":";
  switch (obj.type) {
  case CudaHittableType::HITTABLE_SPHERE:
    oss << "\"SPHERE\",";
    if (obj.sphere) {
      CudaSphere host_sphere;
      cudaMemcpyDeviceToHostSafe(&host_sphere, obj.sphere, 1);
      oss << "\"sphere\":" << cuda_json_sphere(host_sphere);
    } else {
      oss << "\"sphere\":null";
    }
    break;
  case CudaHittableType::HITTABLE_PLANE:
    oss << "\"PLANE\",";
    if (obj.plane) {
      CudaPlane host_plane;
      cudaMemcpyDeviceToHostSafe(&host_plane, obj.plane, 1);
      oss << "\"plane\":" << cuda_json_plane(host_plane);
    } else {
      oss << "\"plane\":null";
    }
    break;
  case CudaHittableType::HITTABLE_BVH_NODE:
    oss << "\"BVH_NODE\",";
    if (obj.bvh_node) {
      CudaBVHNode host_bvh_node;
      cudaMemcpyDeviceToHostSafe(&host_bvh_node, obj.bvh_node, 1);
      oss << "\"bvh_node\":" << cuda_json_bvh_node(host_bvh_node);
    } else {
      oss << "\"bvh_node\":null";
    }
    break;
  case CudaHittableType::HITTABLE_CONSTANT_MEDIUM:
    oss << "\"CONSTANT_MEDIUM\",";
    if (obj.constant_medium) {
      CudaConstantMedium host_constant_medium;
      cudaMemcpyDeviceToHostSafe(&host_constant_medium, obj.constant_medium, 1);
      oss << "\"constant_medium\":"
          << cuda_json_constant_medium(host_constant_medium);
    } else {
      oss << "\"constant_medium\":null";
    }
    break;
  case CudaHittableType::HITTABLE_ROTATE_Y:
    oss << "\"ROTATE_Y\",";
    if (obj.rotate_y) {
      CudaRotateY host_rotate_y;
      cudaMemcpyDeviceToHostSafe(&host_rotate_y, obj.rotate_y, 1);
      oss << "\"rotate_y\":" << cuda_json_rotate_y(host_rotate_y);
    } else {
      oss << "\"rotate_y\":null";
    }
    break;
  case CudaHittableType::HITTABLE_TRANSLATE:
    oss << "\"TRANSLATE\",";
    if (obj.translate) {
      CudaTranslate host_translate;
      cudaMemcpyDeviceToHostSafe(&host_translate, obj.translate, 1);
      oss << "\"translate\":" << cuda_json_translate(host_translate);
    } else {
      oss << "\"translate\":null";
    }
    break;
  case CudaHittableType::HITTABLE_LIST:
    oss << "\"LIST\",";
    if (obj.hittable_list) {
      CudaHittableList host_hittable_list;
      cudaMemcpyDeviceToHostSafe(&host_hittable_list, obj.hittable_list, 1);
      oss << "\"hittable_list\":"
          << cuda_json_hittable_list(host_hittable_list);
    } else {
      oss << "\"hittable_list\":null";
    }
    break;
  }
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA