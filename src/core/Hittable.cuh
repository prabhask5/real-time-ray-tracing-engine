#pragma once

#ifdef USE_CUDA

#include "../optimization/AABB.cuh"
#include "../utils/math/Interval.cuh"
#include "../utils/math/Utility.cuh"
#include "HitRecord.cuh"
#include "Ray.cuh"
#include "Vec3Types.cuh"
#include <assert.h>
#include <curand_kernel.h>

// Forward declarations.
struct CudaBVHNode;
struct CudaConstantMedium;
struct CudaPlane;
struct CudaRotateY;
struct CudaSphere;
struct CudaTranslate;
struct CudaHittableList;
struct CudaTexture;

// Enumeration for CUDA Hittable object types (used for manual dispatch).
enum class CudaHittableType {
  HITTABLE_SPHERE,
  HITTABLE_PLANE,
  HITTABLE_BVH_NODE,
  HITTABLE_CONSTANT_MEDIUM,
  HITTABLE_ROTATE_Y,
  HITTABLE_TRANSLATE,
  HITTABLE_LIST,
};

// POD struct for unified CUDA Hittable using manual dispatch pattern.
struct CudaHittable {
  CudaHittableType type;

  union {
    CudaSphere *sphere;
    CudaPlane *plane;
    CudaBVHNode *bvh_node;
    CudaConstantMedium *constant_medium;
    CudaRotateY *rotate_y;
    CudaTranslate *translate;
    CudaHittableList *hittable_list;
  };
};

// Hittable utility functions.
__device__ bool cuda_hittable_hit(const CudaHittable &hittable,
                                  const CudaRay &ray, CudaInterval t_range,
                                  CudaHitRecord &rec, curandState *rand_state);

__device__ double cuda_hittable_pdf_value(const CudaHittable &hittable,
                                          const CudaPoint3 &origin,
                                          const CudaVec3 &direction);

__device__ CudaVec3 cuda_hittable_random(const CudaHittable &hittable,
                                         const CudaPoint3 &origin,
                                         curandState *state);

__device__ CudaAABB
cuda_hittable_get_bounding_box(const CudaHittable &hittable);

// Helper hittable wrapper functions.
__device__ CudaHittable cuda_make_hittable_sphere(const CudaPoint3 &center,
                                                  double radius,
                                                  const CudaMaterial *material);
__device__ CudaHittable cuda_make_hittable_sphere(
    const CudaPoint3 &before_center, const CudaPoint3 &after_center,
    double radius, const CudaMaterial *material);
__device__ CudaHittable cuda_make_hittable_plane(const CudaPoint3 &corner,
                                                 const CudaVec3 &u_side,
                                                 const CudaVec3 &v_side,
                                                 const CudaMaterial *material);
__device__ CudaHittable cuda_make_hittable_box(const CudaPoint3 &a,
                                               const CudaPoint3 &b,
                                               const CudaMaterial &material);
__device__ CudaHittable cuda_make_hittable_bvh_node(CudaHittable *left,
                                                    CudaHittable *right,
                                                    bool is_leaf = false);
__device__ CudaHittable cuda_make_hittable_constant_medium(
    const CudaHittable *boundary, double density, CudaTexture *texture);
__device__ CudaHittable cuda_make_hittable_rotate_y(const CudaHittable *object,
                                                    double angle_degrees);
__device__ CudaHittable cuda_make_hittable_translate(const CudaHittable *object,
                                                     const CudaVec3 &offset);
__device__ CudaHittable cuda_make_hittable_from_list(CudaHittableList *list);

// Include HittableList.cuh after CudaHittable is defined to avoid circular
// dependency.
#include "HittableList.cuh"

#endif // USE_CUDA
