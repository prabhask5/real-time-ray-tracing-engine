#ifdef USE_CUDA

#include "../optimization/BVHNode.cuh"
#include "../scene/mediums/ConstantMedium.cuh"
#include "../scene/objects/Plane.cuh"
#include "../scene/objects/RotateY.cuh"
#include "../scene/objects/Sphere.cuh"
#include "../scene/objects/Translate.cuh"
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

__device__ CudaAABB
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

// Helper constructor functions.

__device__ CudaHittable cuda_make_hittable_sphere(
    const CudaPoint3 &center, double radius, const CudaMaterial *material) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_SPHERE;
  hittable.sphere = new CudaSphere();
  *hittable.sphere = cuda_make_sphere(center, radius, material);
  return hittable;
}

__device__ CudaHittable cuda_make_hittable_sphere(
    const CudaPoint3 &before_center, const CudaPoint3 &after_center,
    double radius, const CudaMaterial *material) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_SPHERE;
  hittable.sphere = new CudaSphere();
  *hittable.sphere =
      cuda_make_sphere(before_center, after_center, radius, material);
  return hittable;
}

__device__ CudaHittable cuda_make_hittable_plane(const CudaPoint3 &corner,
                                                 const CudaVec3 &u_side,
                                                 const CudaVec3 &v_side,
                                                 const CudaMaterial *material) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_PLANE;
  hittable.plane = new CudaPlane();
  *hittable.plane = cuda_make_plane(corner, u_side, v_side, material);
  return hittable;
}

__device__ CudaHittable cuda_make_hittable_box(const CudaPoint3 &a,
                                               const CudaPoint3 &b,
                                               const CudaMaterial &material) {
  CudaPoint3 min =
      cuda_make_vec3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
  CudaPoint3 max =
      cuda_make_vec3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));

  CudaVec3 dx = cuda_make_vec3(max.x - min.x, 0, 0);
  CudaVec3 dy = cuda_make_vec3(0, max.y - min.y, 0);
  CudaVec3 dz = cuda_make_vec3(0, 0, max.z - min.z);

  CudaHittable *box = new CudaHittable[6];
  int i = 0;

  box[i++] = cuda_make_hittable_plane(cuda_make_vec3(min.x, min.y, max.z), dx,
                                      dy, &material);
  box[i++] = cuda_make_hittable_plane(cuda_make_vec3(max.x, min.y, max.z),
                                      cuda_vec3_negate(dz), dy, &material);
  box[i++] = cuda_make_hittable_plane(cuda_make_vec3(max.x, min.y, min.z),
                                      cuda_vec3_negate(dx), dy, &material);
  box[i++] = cuda_make_hittable_plane(cuda_make_vec3(min.x, min.y, min.z), dz,
                                      dy, &material);
  box[i++] = cuda_make_hittable_plane(cuda_make_vec3(min.x, max.y, max.z), dx,
                                      cuda_vec3_negate(dz), &material);
  box[i++] = cuda_make_hittable_plane(cuda_make_vec3(min.x, min.y, min.z), dx,
                                      dz, &material);

  CudaHittableList *list = new CudaHittableList();
  *list = cuda_make_hittable_list(box, 6);
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_LIST;
  hittable.hittable_list = list;
  return hittable;
}

__device__ CudaHittable cuda_make_hittable_bvh_node(CudaHittable *left,
                                                    CudaHittable *right,
                                                    bool is_leaf) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_BVH_NODE;
  hittable.bvh_node = new CudaBVHNode();
  *hittable.bvh_node = cuda_make_bvh_node(left, right, is_leaf);
  return hittable;
}

__device__ CudaHittable cuda_make_hittable_constant_medium(
    const CudaHittable *boundary, double density, CudaTexture *texture) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_CONSTANT_MEDIUM;
  hittable.constant_medium = new CudaConstantMedium();
  *hittable.constant_medium =
      cuda_make_constant_medium(boundary, density, texture);
  return hittable;
}

__device__ CudaHittable cuda_make_hittable_rotate_y(const CudaHittable *object,
                                                    double angle_degrees) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_ROTATE_Y;
  hittable.rotate_y = new CudaRotateY();
  *hittable.rotate_y = cuda_make_rotate_y(object, angle_degrees);
  return hittable;
}

__device__ CudaHittable cuda_make_hittable_translate(const CudaHittable *object,
                                                     const CudaVec3 &offset) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_TRANSLATE;
  hittable.translate = new CudaTranslate();
  *hittable.translate = cuda_make_translate(object, offset);
  return hittable;
}

__device__ CudaHittable cuda_make_hittable_from_list(CudaHittableList *list) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_LIST;
  hittable.hittable_list = list;
  return hittable;
}

#endif // USE_CUDA