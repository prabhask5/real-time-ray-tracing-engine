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

// Forward declarations
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

// Unified CUDA Hittable using manual dispatch pattern.
// Defines an object that can be hit by a light ray.
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

  __device__ CudaHittable() {} // Default constructor.

  // Main interface methods using manual dispatch.
  // This function checks if the ray hits the hittable object with the t values
  // in the interval range ray_t- if true, writes the hit record info in the hit
  // record reference object. NOTE: A ray can be represented through the
  // parametric equation Vec3 point = r.origin() + t * r.direction(), t is how
  // far along the way you are (t is the parameter, time usually), if t = 0
  // we're at the origin and if t = INF we're infinitly far in the direction of
  // the ray. This function takes in an interval on t values to search through
  // to determine if the ray intersected with the hittable during that time.
  __device__ bool hit(const CudaRay &ray, CudaInterval t_range,
                      CudaHitRecord &rec, curandState *rand_state) const;

  __device__ double pdf_value(const CudaPoint3 &origin,
                              const CudaVec3 &direction) const;

  __device__ CudaVec3 random(const CudaPoint3 &origin,
                             curandState *state) const;

  __device__ CudaAABB get_bounding_box() const;
};

// Helper constructor functions.
__device__ CudaHittable cuda_make_static_sphere(const CudaPoint3 &center,
                                                double radius,
                                                const CudaMaterial *material);
__device__ CudaHittable cuda_make_moving_sphere(const CudaPoint3 &before_center,
                                                const CudaPoint3 &after_center,
                                                double radius,
                                                const CudaMaterial *material);
__device__ CudaHittable cuda_make_plane(const CudaPoint3 &corner,
                                        const CudaVec3 &u_side,
                                        const CudaVec3 &v_side,
                                        const CudaMaterial *material);
__device__ CudaHittable cuda_make_box(const CudaPoint3 &a, const CudaPoint3 &b,
                                      const CudaMaterial &material);
__device__ CudaHittable cuda_make_bvh_node(CudaHittable *left,
                                           CudaHittable *right,
                                           bool is_leaf = false);
__device__ CudaHittable cuda_make_constant_medium(const CudaHittable *boundary,
                                                  double density,
                                                  CudaTexture *texture);
__device__ CudaHittable cuda_make_rotate_y(const CudaHittable *object,
                                           double angle_degrees);
__device__ CudaHittable cuda_make_translate(const CudaHittable *object,
                                            const CudaVec3 &offset);
__device__ CudaHittable cuda_make_hittable_list(CudaHittableList *list);

// Include HittableList.cuh after CudaHittable is defined to avoid circular
// dependency.
#include "HittableList.cuh"

#endif // USE_CUDA
