#pragma once

#ifdef USE_CUDA

#include "../optimization/AABB.cuh"
#include "../utils/math/Interval.cuh"
#include "HitRecord.cuh"
#include "Ray.cuh"
#include "Vec3Types.cuh"
#include <curand_kernel.h>

// Forward declarations for hittable structs.
struct CudaSphere;
struct CudaPlane;
struct CudaBVHNode;
struct CudaConstantMedium;
struct CudaRotateY;
struct CudaTranslate;
struct CudaHittableList;

// Enumeration for CUDA Hittable object types (used for manual dispatch).
enum CudaHittableType {
  HITTABLE = 0,
  HITTABLE_SPHERE = 1,
  HITTABLE_PLANE = 2,
  HITTABLE_BVH_NODE = 3,
  HITTABLE_CONSTANT_MEDIUM = 4,
  HITTABLE_ROTATE_Y = 5,
  HITTABLE_TRANSLATE = 6,
  HITTABLE_LIST = 7,
};

// Unified CUDA Hittable using manual dispatch pattern.
struct CudaHittable {
  CudaHittableType type;

  union {
    CudaSphere sphere;
    CudaPlane plane;
    CudaBVHNode bvh_node;
    CudaConstantMedium constant_medium;
    CudaRotateY rotate_y;
    CudaTranslate translate;
    CudaHittableList hittable_list;
  };

  // Main interface methods using manual dispatch.
  __device__ bool hit(const CudaRay &ray, CudaInterval t_range,
                      CudaHitRecord &rec, curandState *rand_state) const {
    switch (type) {
    case CudaHittableType::HITTABLE_SPHERE:
      return sphere.hit(ray, t_range, rec, rand_state);
    case CudaHittableType::HITTABLE_PLANE:
      return plane.hit(ray, t_range, rec, rand_state);
    case CudaHittableType::HITTABLE_BVH_NODE:
      return bvh_node.hit(ray, t_range, rec, rand_state);
    case CudaHittableType::HITTABLE_CONSTANT_MEDIUM:
      constant_medium.hit(ray, t_range, rec, rand_state);
    case CudaHittableType::HITTABLE_ROTATE_Y:
      return rotate_y.hit(ray, t_range, rec, rand_state);
    case CudaHittableType::HITTABLE_TRANSLATE:
      return translate.hit(ray, t_range, rec, rand_state);
    case CudaHittableType::HITTABLE_LIST:
      return hittable_list.hit(ray, t_range, rec, rand_state);
    default:
      return false;
    }
  }

  __device__ double pdf_value(const CudaPoint3 &origin,
                              const CudaVec3 &direction) const {
    switch (type) {
    case CudaHittableType::HITTABLE_SPHERE:
      return sphere.pdf_value(origin, direction);
    case CudaHittableType::HITTABLE_PLANE:
      return plane.pdf_value(origin, direction);
    case CudaHittableType::HITTABLE_BVH_NODE:
      return bvh_node.pdf_value(origin, direction);
    case CudaHittableType::HITTABLE_LIST:
      return hittable_list.pdf_value(origin, direction);
    default:
      return 0.0;
    }
  }

  __device__ CudaVec3 random(const CudaPoint3 &origin,
                             curandState *state) const {
    switch (type) {
    case CudaHittableType::HITTABLE_SPHERE:
      return sphere.random(origin, state);
    case CudaHittableType::HITTABLE_PLANE:
      return plane.random(origin, state);
    case CudaHittableType::HITTABLE_BVH_NODE:
      return bvh_node.random(origin, state);
    case CudaHittableType::HITTABLE_LIST:
      return hittable_list.random(origin, state);
    default:
      return CudaVec3(1, 0, 0);
    }
  }

  __device__ CudaAABB get_bounding_box() const {
    switch (type) {
    case CudaHittableType::HITTABLE_SPHERE:
      return sphere.get_bounding_box();
    case CudaHittableType::HITTABLE_PLANE:
      return plane.get_bounding_box();
    case CudaHittableType::HITTABLE_BVH_NODE:
      return bvh_node.get_bounding_box();
    case CudaHittableType::HITTABLE_CONSTANT_MEDIUM:
      return constant_medium.get_bounding_box();
    case CudaHittableType::HITTABLE_ROTATE_Y:
      return rotate_y.get_bounding_box();
    case CudaHittableType::HITTABLE_TRANSLATE:
      return translate.get_bounding_box();
    case CudaHittableType::HITTABLE_LIST:
      return hittable_list.get_bounding_box();
    default:
      return CudaAABB();
    }
  }
};

#endif // USE_CUDA
