#pragma once

#ifdef USE_CUDA

#include "../../core/AABB.cuh"
#include "../../core/HitRecord.cuh"
#include "../../core/Interval.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.hpp"
#include "../materials/CudaMaterial.cuh"
#include "../math/ONB.cuh"
#include <curand_kernel.h>

// CUDA-compatible sphere with motion blur support.
struct CudaSphere {
  CudaPoint3 center_start;
  CudaVec3 velocity;
  double radius;
  CudaMaterial material;
  CudaAABB bbox;
};

// Compute center of sphere at time t
__device__ inline CudaPoint3 cuda_sphere_center(const CudaSphere &sphere,
                                                double time) {
  return sphere.center_start + time * sphere.velocity;
}

// Initialize sphere with static center
__host__ __device__ inline void cuda_init_sphere(CudaSphere &s,
                                                 const CudaPoint3 &center,
                                                 double radius,
                                                 const CudaMaterial &material) {

  s.center_start = center;
  s.velocity = CudaVec3(0, 0, 0);
  s.radius = fmax(0.0, radius);
  s.material = material;

  CudaVec3 r(radius, radius, radius);
  s.bbox = CudaAABB(center - r, center + r);
}

// Initialize sphere with moving center (motion blur)
__host__ __device__ inline void
cuda_init_sphere_motion(CudaSphere &s, const CudaPoint3 &start,
                        const CudaPoint3 &end, double radius,
                        const CudaMaterial &material) {

  s.center_start = start;
  s.velocity = end - start;
  s.radius = fmax(0.0, radius);
  s.material = material;

  CudaVec3 r(radius, radius, radius);
  CudaAABB box1(start - r, start + r);
  CudaAABB box2(end - r, end + r);
  s.bbox = CudaAABB(box1, box2);
}

// Hit test for sphere
__device__ inline bool cuda_sphere_hit(const CudaSphere &s, const CudaRay &ray,
                                       CudaInterval t_range,
                                       CudaHitRecord &rec) {

  CudaPoint3 center = cuda_sphere_center(s, ray.time);
  CudaVec3 oc = center - ray.origin;
  double a = cuda_length_squared(ray.direction);
  double h = cuda_dot_product(ray.direction, oc);
  double c = cuda_length_squared(oc) - s.radius * s.radius;

  double discriminant = h * h - a * c;
  if (discriminant < 0.0)
    return false;
  double sqrt_d = sqrt(discriminant);

  double root = (h - sqrt_d) / a;
  if (!t_range.surrounds(root)) {
    root = (h + sqrt_d) / a;
    if (!t_range.surrounds(root))
      return false;
  }

  rec.t = root;
  rec.point = cuda_ray_at(ray, rec.t);
  CudaVec3 outward = (rec.point - center) / s.radius;
  rec.normal = outward;
  rec.front_face = cuda_dot_product(ray.direction, outward) < 0;
  rec.material = s.material;

  // Spherical coordinates (u, v)
  double theta = acos(-outward.y());
  double phi = atan2(-outward.z(), outward.x()) + CUDA_PI;
  rec.u = phi / (2 * CUDA_PI);
  rec.v = theta / CUDA_PI;

  return true;
}

// PDF for sampling the sphere (solid angle)
__device__ inline double cuda_sphere_pdf_value(const CudaSphere &s,
                                               const CudaPoint3 &origin,
                                               const CudaVec3 &direction) {

  CudaRay ray(origin, direction);
  CudaHitRecord temp;
  if (!cuda_sphere_hit(s, ray, CudaInterval(0.001, CUDA_INF), temp))
    return 0.0;

  double dist2 = cuda_length_squared(s.center_start - origin);
  double cos_theta_max = sqrt(1 - s.radius * s.radius / dist2);
  double solid_angle = 2 * CUDA_PI * (1 - cos_theta_max);
  return 1.0 / solid_angle;
}

// Importance sample a direction toward the sphere
__device__ inline CudaVec3 cuda_sphere_random(const CudaSphere &s,
                                              const CudaPoint3 &origin,
                                              curandState *rand_state) {

  CudaVec3 dir = s.center_start - origin;
  double dist2 = cuda_length_squared(dir);
  CudaONB uvw = cuda_make_onb(dir);

  double r1 = curand_uniform_double(rand_state);
  double r2 = curand_uniform_double(rand_state);

  double z = 1 + r2 * (sqrt(1 - s.radius * s.radius / dist2) - 1);
  double phi = 2 * CUDA_PI * r1;
  double x = cos(phi) * sqrt(1 - z * z);
  double y = sin(phi) * sqrt(1 - z * z);

  return cuda_onb_transform(uvw, CudaVec3(x, y, z));
}

#endif // USE_CUDA
