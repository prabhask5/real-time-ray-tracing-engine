#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Interval.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.hpp"
#include "../materials/CudaMaterial.cuh"
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
};

// Computes if a ray hits a CUDA plane, and writes to record if so.
__device__ bool cuda_plane_hit(const CudaPlane &plane, const CudaRay &ray,
                               CudaInterval t_values, CudaHitRecord &record) {

  double denom = cuda_dot_product(plane.normal, ray.direction);
  if (fabs(denom) < 1e-8)
    return false;

  double t = (plane.D - cuda_dot_product(plane.normal, ray.origin)) / denom;
  if (!t_values.contains(t))
    return false;

  CudaPoint3 intersection = cuda_ray_at(ray, t);
  CudaVec3 planar_hit = intersection - plane.corner;

  double alpha =
      cuda_dot_product(plane.w, cuda_cross_product(planar_hit, plane.v_side));
  double beta =
      cuda_dot_product(plane.w, cuda_cross_product(plane.u_side, planar_hit));

  if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1)
    return false;

  record.t = t;
  record.point = intersection;
  record.normal = plane.normal;
  record.front_face = cuda_dot_product(ray.direction, plane.normal) < 0;
  record.material = plane.material;
  record.u = alpha;
  record.v = beta;

  return true;
}

// Computes the PDF value of sampling the plane from a point along a direction.
__device__ double cuda_plane_pdf_value(const CudaPlane &plane,
                                       const CudaPoint3 &origin,
                                       const CudaVec3 &direction) {

  CudaRay ray(origin, direction);
  CudaHitRecord record;
  if (!cuda_plane_hit(plane, ray, CudaInterval(0.001, CUDA_INF), record))
    return 0.0;

  double distance_squared =
      record.t * record.t * cuda_length_squared(direction);
  double cosine =
      fabs(cuda_dot_product(direction, record.normal) / cuda_length(direction));

  return distance_squared / (cosine * plane.surface_area);
}

// Samples a random direction toward a point on the plane.
__device__ CudaVec3 cuda_plane_random(const CudaPlane &plane,
                                      const CudaPoint3 &origin,
                                      curandState *rand_state) {

  double u = curand_uniform_double(rand_state);
  double v = curand_uniform_double(rand_state);

  CudaPoint3 point_on_plane =
      plane.corner + u * plane.u_side + v * plane.v_side;
  return point_on_plane - origin;
}

// Initialize CUDA plane
__host__ __device__ inline void cuda_init_plane(CudaPlane &dst,
                                                const CudaPoint3 &corner,
                                                const CudaVec3 &u_side,
                                                const CudaVec3 &v_side,
                                                const CudaMaterial &material) {

  dst.corner = corner;
  dst.u_side = u_side;
  dst.v_side = v_side;
  dst.material = material;

  CudaVec3 n = cuda_cross_product(u_side, v_side);
  dst.normal = cuda_unit_vector(n);
  dst.D = cuda_dot_product(dst.normal, corner);
  dst.w = n / cuda_dot_product(n, n);
  dst.surface_area = cuda_length(n);

  CudaPoint3 p0 = corner;
  CudaPoint3 p1 = corner + u_side;
  CudaPoint3 p2 = corner + v_side;
  CudaPoint3 p3 = corner + u_side + v_side;

  CudaAABB box1 = CudaAABB(p0, p3);
  CudaAABB box2 = CudaAABB(p1, p2);
  dst.bbox = CudaAABB(box1, box2);
}

#endif // USE_CUDA
