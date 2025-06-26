#ifdef USE_CUDA

#include "Sphere.cuh"

CudaSphere cuda_make_sphere(const CudaPoint3 &center_pos, double radius,
                            size_t material_index) {
  CudaSphere sphere;
  sphere.center = cuda_make_ray(center_pos, cuda_make_vec3(0, 0, 0));
  sphere.radius = fmax(0.0, radius);
  sphere.material_index = material_index;
  CudaVec3 r = cuda_make_vec3(radius, radius, radius);
  sphere.bbox = cuda_make_aabb(cuda_vec3_subtract(center_pos, r),
                               cuda_vec3_add(center_pos, r));
  return sphere;
}

CudaSphere cuda_make_sphere(const CudaPoint3 &before_center,
                            const CudaPoint3 &after_center, double radius,
                            size_t material_index) {
  CudaSphere sphere;
  sphere.center = cuda_make_ray(
      before_center, cuda_vec3_subtract(after_center, before_center));
  sphere.radius = fmax(0.0, radius);
  sphere.material_index = material_index;
  CudaVec3 r = cuda_make_vec3(radius, radius, radius);
  CudaAABB box1 = cuda_make_aabb(cuda_vec3_subtract(before_center, r),
                                 cuda_vec3_add(before_center, r));
  CudaAABB box2 = cuda_make_aabb(cuda_vec3_subtract(after_center, r),
                                 cuda_vec3_add(after_center, r));
  sphere.bbox = cuda_make_aabb(box1, box2);
  return sphere;
}

__device__ bool cuda_sphere_hit(const CudaSphere &sphere, const CudaRay &ray,
                                CudaInterval t_range, CudaHitRecord &rec,
                                curandState *rand_state) {
  CudaPoint3 current_center = cuda_ray_at(sphere.center, ray.time);
  CudaVec3 oc = cuda_vec3_subtract(current_center, ray.origin);

  double a = cuda_vec3_length_squared(ray.direction);
  double h = cuda_vec3_dot_product(ray.direction, oc);
  double c = cuda_vec3_length_squared(oc) - sphere.radius * sphere.radius;

  double discriminant = h * h - a * c;
  if (discriminant < 0.0)
    return false;

  double sqrt_d = sqrt(discriminant);

  double root = (h - sqrt_d) / a;
  if (!cuda_interval_surrounds(t_range, root)) {
    root = (h + sqrt_d) / a;
    if (!cuda_interval_surrounds(t_range, root))
      return false;
  }

  rec.t = root;
  rec.point = cuda_ray_at(ray, rec.t);
  CudaVec3 outward = cuda_vec3_divide_scalar(
      cuda_vec3_subtract(rec.point, current_center), sphere.radius);
  cuda_hit_record_set_face_normal(rec, ray, outward);
  rec.material_index = sphere.material_index;

  double theta = acos(-outward.y);
  double phi = atan2(-outward.z, outward.x) + CUDA_PI;

  rec.u = phi / (2 * CUDA_PI);
  rec.v = theta / CUDA_PI;

  return true;
}

__device__ double cuda_sphere_pdf_value(const CudaSphere &sphere,
                                        const CudaPoint3 &origin,
                                        const CudaVec3 &direction) {
  CudaRay ray = cuda_make_ray(origin, direction);
  CudaHitRecord temp;
  if (!cuda_sphere_hit(sphere, ray, cuda_make_interval(0.001, CUDA_INF), temp,
                       nullptr))
    return 0.0;

  double dist2 = cuda_vec3_length_squared(
      cuda_vec3_subtract(cuda_ray_at(sphere.center, 0), origin));
  double cos_theta_max = sqrt(1 - sphere.radius * sphere.radius / dist2);
  double solid_angle = 2 * CUDA_PI * (1 - cos_theta_max);

  return 1.0 / solid_angle;
}

__device__ CudaVec3 cuda_sphere_random(const CudaSphere &sphere,
                                       const CudaPoint3 &origin,
                                       curandState *rand_state) {
  CudaVec3 dir = cuda_vec3_subtract(cuda_ray_at(sphere.center, 0), origin);
  double dist2 = cuda_vec3_length_squared(dir);
  CudaONB uvw = cuda_make_onb(dir);

  double r1 = cuda_random_double(rand_state);
  double r2 = cuda_random_double(rand_state);
  double z = 1 + r2 * (sqrt(1 - sphere.radius * sphere.radius / dist2) - 1);
  double phi = 2 * CUDA_PI * r1;
  double x = cos(phi) * sqrt(1 - z * z);
  double y = sin(phi) * sqrt(1 - z * z);

  return cuda_onb_transform(uvw, cuda_make_vec3(x, y, z));
}

#endif // USE_CUDA