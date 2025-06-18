#pragma once

#ifdef USE_CUDA

#include "../../core/HitRecord.cuh"
#include "../../core/Ray.cuh"
#include "../../core/Vec3Types.cuh"
#include "../../optimization/AABB.cuh"
#include "../../utils/math/Interval.cuh"
#include "../../utils/math/ONB.cuh"
#include "../../utils/math/Vec3Utility.cuh"
#include "../materials/Material.cuh"
#include <curand_kernel.h>

// CUDA-compatible sphere with motion blur support.
struct CudaSphere {
  CudaRay center;
  double radius;
  CudaMaterial material;
  CudaAABB bbox;

  // Initialize sphere with static center.
  __device__ CudaSphere(const CudaPoint3 &_center, double _radius,
                        const CudaMaterial &_material)
      : center(_center, CudaVec3(0, 0, 0)), radius(fmax(0.0, _radius)),
        material(_material) {
    CudaVec3 r(radius, radius, radius);
    bbox = CudaAABB(_center - r, _center + r);
  }

  // Initialize sphere with moving center (motion blur).
  __device__ CudaSphere(const Point3 &before_center, const Point3 &after_center,
                        double _radius, MaterialPtr _material)
      : center(before_center, after_center - before_center),
        radius(fmax(0.0, _radius)), material(_material) {
    CudaVec3 r(radius, radius, radius);
    CudaAABB box1(before_center - r, before_center + r);
    CudaAABB box2(after_center - r, after_center + r);
    bbox = CudaAABB(box1, box2);
  }

  // Hit test for sphere.
  __device__ inline bool hit(const CudaRay &ray, CudaInterval t_range,
                             CudaHitRecord &rec, curandState *rand_state) {
    CudaPoint3 current_center = center.at(ray.time);
    CudaVec3 oc = current_center - ray.origin;
    double a = ray.direction.length_squared();
    double h = cuda_dot_product(ray.direction, oc);
    double c = oc.length_squared() - radius * radius;

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
    rec.point = ray.at(rec.t);
    CudaVec3 outward = (rec.point - current_center) / radius;
    cuda_set_face_normal(rec, ray, outward);
    rec.material_type = material.type;
    rec.material_data = (void *)&material;

    // Spherical coordinates (u, v).
    double theta = acos(-outward.y);
    double phi = atan2(-outward.z, outward.x) + CUDA_PI;
    rec.u = phi / (2 * CUDA_PI);
    rec.v = theta / CUDA_PI;

    return true;
  }

  // PDF for sampling the sphere (solid angle).
  __device__ inline double pdf_value(const CudaPoint3 &origin,
                                     const CudaVec3 &direction) {
    CudaRay ray = CudaRay(origin, direction);
    CudaHitRecord temp;
    if (!this->hit(ray, CudaInterval(0.001, CUDA_INF), temp, nullptr))
      return 0.0;

    double dist2 = (center.at(0) - origin).length_squared();
    double cos_theta_max = sqrt(1 - radius * radius / dist2);
    double solid_angle = 2 * CUDA_PI * (1 - cos_theta_max);
    return 1.0 / solid_angle;
  }

  // Importance sample a direction toward the sphere.
  __device__ inline CudaVec3 random(const CudaPoint3 &origin,
                                    curandState *rand_state) {

    CudaVec3 dir = center.at(0) - origin;
    double dist2 = dir.length_squared();
    CudaONB uvw(dir);

    double r1 = curand_uniform_double(rand_state);
    double r2 = curand_uniform_double(rand_state);

    double z = 1 + r2 * (sqrt(1 - radius * radius / dist2) - 1);
    double phi = 2 * CUDA_PI * r1;
    double x = cos(phi) * sqrt(1 - z * z);
    double y = sin(phi) * sqrt(1 - z * z);

    return uvw.transform(CudaVec3(x, y, z));
  }

  // Get bounding box for the sphere.
  __device__ inline CudaAABB get_bounding_box() { return bbox; }
};

#endif // USE_CUDA
