#ifdef USE_CUDA

#include "Sphere.cuh"

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
  CudaRay ray = cuda_make_ray(origin, direction, 0.0);
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