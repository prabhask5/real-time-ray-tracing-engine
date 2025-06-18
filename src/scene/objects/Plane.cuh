#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../optimization/AABB.cuh"
#include "../../utils/math/Interval.cuh"
#include "../../utils/math/Vec3Utility.cuh"
#include "../materials/Material.cuh"
#include <curand_kernel.h>

// A planar parallelogram with two edge vectors and a corner point.
struct CudaPlane {
  CudaPoint3 corner;
  CudaVec3 u_side;
  CudaVec3 v_side;
  CudaVec3 w; // Used for computing UV coordinates
  CudaVec3 normal;
  double D;
  double surface_area;
  CudaAABB bbox;
  CudaMaterial material;

  // Initialize CUDA plane.
  __device__ CudaPlane(const CudaPoint3 &_corner, const CudaVec3 &_u_side,
                       const CudaVec3 &_v_side, const CudaMaterial &_material)
      : corner(_corner), u_side(_u_side), v_side(_v_side), material(_material) {
    CudaVec3 n = cuda_cross_product(u_side, v_side);
    normal = cuda_unit_vector(n);
    D = cuda_dot_product(normal, corner);
    w = n / cuda_dot_product(n, n);
    surface_area = n.length();

    CudaPoint3 p0 = corner;
    CudaPoint3 p1 = corner + u_side;
    CudaPoint3 p2 = corner + v_side;
    CudaPoint3 p3 = corner + u_side + v_side;

    CudaAABB box1 = CudaAABB(p0, p3);
    CudaAABB box2 = CudaAABB(p1, p2);
    bbox = CudaAABB(box1, box2);
  }

  // Computes if a ray hits a CUDA plane, and writes to record if so.
  __device__ bool hit(const CudaRay &ray, CudaInterval t_values,
                      CudaHitRecord &record, curandState *rand_state) {

    double denom = cuda_dot_product(normal, ray.direction);
    if (fabs(denom) < 1e-8)
      return false;

    double t = (D - cuda_dot_product(normal, ray.origin)) / denom;
    if (!t_values.contains(t))
      return false;

    CudaPoint3 intersection = ray.at(t);
    CudaVec3 planar_hit = intersection - corner;

    double alpha = cuda_dot_product(w, cuda_cross_product(planar_hit, v_side));
    double beta = cuda_dot_product(w, cuda_cross_product(u_side, planar_hit));

    if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1)
      return false;

    record.t = t;
    record.point = intersection;
    cuda_set_face_normal(record, ray, normal);
    record.material_type = material.type;
    record.material_data = (void *)&material;
    record.u = alpha;
    record.v = beta;

    return true;
  }

  // Computes the PDF value of sampling the plane from a point along a
  // direction.
  __device__ double pdf_value(const CudaPoint3 &origin,
                              const CudaVec3 &direction) {

    CudaRay ray = CudaRay(origin, direction);
    CudaHitRecord record;
    if (!plane.hit(ray, CudaInterval(0.001, CUDA_INF), record))
      return 0.0;

    double distance_squared = record.t * record.t * direction.length_squared();
    double cosine =
        fabs(cuda_dot_product(direction, record.normal) / direction.length());

    return distance_squared / (cosine * surface_area);
  }

  // Samples a random direction toward a point on the plane.
  __device__ CudaVec3 random(const CudaPoint3 &origin,
                             curandState *rand_state) {

    double u = curand_uniform_double(rand_state);
    double v = curand_uniform_double(rand_state);

    CudaPoint3 point_on_plane = corner + u * u_side + v * v_side;
    return point_on_plane - origin;
  }

  // Get bounding box for the plane.
  __device__ inline CudaAABB get_bounding_box() { return bbox; }
};

#endif // USE_CUDA
