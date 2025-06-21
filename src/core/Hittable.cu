#ifdef USE_CUDA

#include "../optimization/BVHNode.cuh"
#include "../scene/mediums/ConstantMedium.cuh"
#include "../scene/objects/Plane.cuh"
#include "../scene/objects/RotateY.cuh"
#include "../scene/objects/Sphere.cuh"
#include "../scene/objects/Translate.cuh"
#include "Hittable.cuh"
#include "HittableList.cuh"

// CudaHittable constructors and dispatch methods implementation

__device__ bool CudaHittable::hit(const CudaRay &ray, CudaInterval t_values,
                                  CudaHitRecord &record,
                                  curandState *rand_state) const {
  switch (type) {
  case CudaHittableType::HITTABLE_SPHERE:
    return sphere->hit(ray, t_values, record, rand_state);
  case CudaHittableType::HITTABLE_PLANE:
    return plane->hit(ray, t_values, record, rand_state);
  case CudaHittableType::HITTABLE_BVH_NODE:
    return bvh_node->hit(ray, t_values, record, rand_state);
  case CudaHittableType::HITTABLE_CONSTANT_MEDIUM:
    return constant_medium->hit(ray, t_values, record, rand_state);
  case CudaHittableType::HITTABLE_ROTATE_Y:
    return rotate_y->hit(ray, t_values, record, rand_state);
  case CudaHittableType::HITTABLE_TRANSLATE:
    return translate->hit(ray, t_values, record, rand_state);
  case CudaHittableType::HITTABLE_LIST:
    return hittable_list->hit(ray, t_values, record, rand_state);
  default:
    return false;
  }
}

__device__ double CudaHittable::pdf_value(const CudaPoint3 &origin,
                                          const CudaVec3 &direction) const {
  switch (type) {
  case CudaHittableType::HITTABLE_SPHERE:
    return sphere->pdf_value(origin, direction);
  case CudaHittableType::HITTABLE_PLANE:
    return plane->pdf_value(origin, direction);
  case CudaHittableType::HITTABLE_BVH_NODE:
    return bvh_node->pdf_value(origin, direction);
  case CudaHittableType::HITTABLE_CONSTANT_MEDIUM:
    return 0.0; // ConstantMedium doesn't implement pdf_value
  case CudaHittableType::HITTABLE_ROTATE_Y:
    return rotate_y->pdf_value(origin, direction);
  case CudaHittableType::HITTABLE_TRANSLATE:
    return translate->pdf_value(origin, direction);
  case CudaHittableType::HITTABLE_LIST:
    return hittable_list->pdf_value(origin, direction);
  default:
    return 0.0;
  }
}

__device__ CudaVec3 CudaHittable::random(const CudaPoint3 &origin,
                                         curandState *state) const {
  switch (type) {
  case CudaHittableType::HITTABLE_SPHERE:
    return sphere->random(origin, state);
  case CudaHittableType::HITTABLE_PLANE:
    return plane->random(origin, state);
  case CudaHittableType::HITTABLE_BVH_NODE:
    return bvh_node->random(origin, state);
  case CudaHittableType::HITTABLE_CONSTANT_MEDIUM:
    return CudaVec3(1, 0, 0); // ConstantMedium doesn't implement random
  case CudaHittableType::HITTABLE_ROTATE_Y:
    return rotate_y->random(origin, state);
  case CudaHittableType::HITTABLE_TRANSLATE:
    return translate->random(origin, state);
  case CudaHittableType::HITTABLE_LIST:
    return hittable_list->random(origin, state);
  default:
    return CudaVec3(1, 0, 0);
  }
}

__device__ CudaAABB CudaHittable::get_bounding_box() const {
  switch (type) {
  case CudaHittableType::HITTABLE_SPHERE:
    return sphere->get_bounding_box();
  case CudaHittableType::HITTABLE_PLANE:
    return plane->get_bounding_box();
  case CudaHittableType::HITTABLE_BVH_NODE:
    return bvh_node->get_bounding_box();
  case CudaHittableType::HITTABLE_CONSTANT_MEDIUM:
    return constant_medium->get_bounding_box();
  case CudaHittableType::HITTABLE_ROTATE_Y:
    return rotate_y->get_bounding_box();
  case CudaHittableType::HITTABLE_TRANSLATE:
    return translate->get_bounding_box();
  case CudaHittableType::HITTABLE_LIST:
    return hittable_list->get_bounding_box();
  default:
    return CudaAABB();
  }
}

// Helper constructor functions.

__device__ CudaHittable cuda_make_static_sphere(const CudaPoint3 &center,
                                                double radius,
                                                const CudaMaterial *material) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_SPHERE;
  hittable.sphere = new CudaSphere(center, radius, material);
  return hittable;
}

__device__ CudaHittable cuda_make_moving_sphere(const CudaPoint3 &before_center,
                                                const CudaPoint3 &after_center,
                                                double radius,
                                                const CudaMaterial *material) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_SPHERE;
  hittable.sphere =
      new CudaSphere(before_center, after_center, radius, material);
  return hittable;
}

__device__ CudaHittable cuda_make_plane(const CudaPoint3 &corner,
                                        const CudaVec3 &u_side,
                                        const CudaVec3 &v_side,
                                        const CudaMaterial *material) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_PLANE;
  hittable.plane = new CudaPlane(corner, u_side, v_side, material);
  return hittable;
}

__device__ CudaHittable cuda_make_box(const CudaPoint3 &a, const CudaPoint3 &b,
                                      const CudaMaterial &material) {
  CudaPoint3 min(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
  CudaPoint3 max(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));

  CudaVec3 dx(max.x - min.x, 0, 0);
  CudaVec3 dy(0, max.y - min.y, 0);
  CudaVec3 dz(0, 0, max.z - min.z);

  CudaHittable box[6];
  int i = 0;

  // +Z (front).
  box[i++] =
      cuda_make_plane(CudaPoint3(min.x, min.y, max.z), dx, dy, &material);

  // +X (right).
  box[i++] =
      cuda_make_plane(CudaPoint3(max.x, min.y, max.z), -dz, dy, &material);

  // -Z (back).
  box[i++] =
      cuda_make_plane(CudaPoint3(max.x, min.y, min.z), -dx, dy, &material);

  // -X (left).
  box[i++] =
      cuda_make_plane(CudaPoint3(min.x, min.y, min.z), dz, dy, &material);

  // +Y (top).
  box[i++] =
      cuda_make_plane(CudaPoint3(min.x, max.y, max.z), dx, -dz, &material);

  // -Y (bottom).
  box[i++] =
      cuda_make_plane(CudaPoint3(min.x, min.y, min.z), dx, dz, &material);

  // Create a hittable list on the heap.
  CudaHittableList *list = new CudaHittableList(box, 6);
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_LIST;
  hittable.hittable_list = list;
  return hittable;
}

__device__ CudaHittable cuda_make_bvh_node(CudaHittable *left,
                                           CudaHittable *right, bool is_leaf) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_BVH_NODE;
  hittable.bvh_node = new CudaBVHNode(left, right, is_leaf);
  return hittable;
}

__device__ CudaHittable cuda_make_constant_medium(const CudaHittable *boundary,
                                                  double density,
                                                  CudaTexture *texture) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_CONSTANT_MEDIUM;
  hittable.constant_medium = new CudaConstantMedium(boundary, density, texture);
  return hittable;
}

__device__ CudaHittable cuda_make_rotate_y(const CudaHittable *object,
                                           double angle_degrees) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_ROTATE_Y;
  hittable.rotate_y = new CudaRotateY(object, angle_degrees);
  return hittable;
}

__device__ CudaHittable cuda_make_translate(const CudaHittable *object,
                                            const CudaVec3 &offset) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_TRANSLATE;
  hittable.translate = new CudaTranslate(object, offset);
  return hittable;
}

__device__ CudaHittable cuda_make_hittable_list(CudaHittableList *list) {
  CudaHittable hittable;
  hittable.type = CudaHittableType::HITTABLE_LIST;
  hittable.hittable_list = list;
  return hittable;
}

#endif // USE_CUDA