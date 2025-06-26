#ifdef USE_CUDA

#include "../optimization/BVHNode.cuh"
#include "../scene/mediums/ConstantMedium.cuh"
#include "../scene/objects/Plane.cuh"
#include "../scene/objects/RotateY.cuh"
#include "../scene/objects/Sphere.cuh"
#include "../scene/objects/Translate.cuh"
#include "../utils/memory/CudaSceneContext.cuh"
#include "Hittable.cuh"
#include "HittableList.cuh"

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

#endif // USE_CUDA