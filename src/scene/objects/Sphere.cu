#ifdef USE_CUDA

#include "Sphere.cuh"

__device__ CudaSphere::CudaSphere(const CudaPoint3 &_center, double _radius,
                                  const CudaMaterial *_material)
    : center(_center, CudaVec3(0, 0, 0)), radius(fmax(0.0, _radius)),
      material(_material) {
  CudaVec3 r(radius, radius, radius);
  bbox = CudaAABB(_center - r, _center + r);
}

__device__ CudaSphere::CudaSphere(const CudaPoint3 &before_center,
                                  const CudaPoint3 &after_center,
                                  double _radius, const CudaMaterial *_material)
    : center(before_center, after_center - before_center),
      radius(fmax(0.0, _radius)), material(_material) {
  CudaVec3 r(radius, radius, radius);
  CudaAABB box1(before_center - r, before_center + r);
  CudaAABB box2(after_center - r, after_center + r);
  bbox = CudaAABB(box1, box2);
}

__device__ bool CudaSphere::hit(const CudaRay &ray, CudaInterval t_range,
                                CudaHitRecord &rec,
                                curandState *rand_state) const {
  // Let the ray be: r(t) = origin + t * direction
  // We want to find t such that: [r(t) - center]^2 = radius^2
  // This yields a quadratic: a*t^2 - 2h*t + c = 0
  // Where: oc = center - origin
  // a = direction.length_squared()
  // h = dot(direction, oc)
  // c = [oc]^2 - r^2
  CudaPoint3 current_center = center.at(ray.time);
  CudaVec3 oc = current_center - ray.origin;

  double a = ray.direction.length_squared();
  double h = cuda_dot_product(ray.direction, oc);
  double c = oc.length_squared() - radius * radius;

  // Find the discriminant to easily see if there's any intersection points.
  double discriminant = h * h - a * c;
  if (discriminant < 0.0)
    return false;

  double sqrt_d = sqrt(discriminant);

  // Find the nearest root that lies in the acceptable range.
  double root = (h - sqrt_d) / a; // Via the quadratic equation.
  if (!t_range.surrounds(root)) {
    root = (h + sqrt_d) / a; // Other root.
    if (!t_range.surrounds(root))
      return false; // Seems like none exist then.
  }

  // Populate the hit record.
  rec.t = root;
  rec.point = ray.at(rec.t);
  CudaVec3 outward = (rec.point - current_center) / radius;
  cuda_set_face_normal(rec, ray, outward);
  rec.material_pointer = const_cast<CudaMaterial *>(material);

  // outward_normal: a given point on the sphere of radius one, centered at
  // the origin. record.u: returned value [0,1] of angle around the Y axis
  // from X=-1. record.v: returned value [0,1] of angle from Y=-1 to Y=+1.
  //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
  //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
  //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

  double theta = acos(-outward.y);
  double phi = atan2(-outward.z, outward.x) + CUDA_PI;

  rec.u = phi / (2 * CUDA_PI);
  rec.v = theta / CUDA_PI;

  return true;
}

__device__ double CudaSphere::pdf_value(const CudaPoint3 &origin,
                                        const CudaVec3 &direction) const {
  // This method only works for stationary spheres.

  CudaRay ray = CudaRay(origin, direction);
  CudaHitRecord temp;
  if (!this->hit(ray, CudaInterval(0.001, CUDA_INF), temp, nullptr))
    return 0.0;

  double dist2 = (center.at(0) - origin).length_squared();
  double cos_theta_max = sqrt(1 - radius * radius / dist2);
  double solid_angle = 2 * CUDA_PI * (1 - cos_theta_max);

  // PDF formula for solid angle sampling.
  return 1.0 / solid_angle;
}

__device__ CudaVec3 CudaSphere::random(const CudaPoint3 &origin,
                                       curandState *rand_state) const {

  CudaVec3 dir = center.at(0) - origin;
  double dist2 = dir.length_squared();
  CudaONB uvw(dir);

  // Generates a random direction vector that points toward a sphere from a
  // point in space that is outside the sphere. This is used for a sphere to
  // sample on directions that actually intersect the sphere — this is faster
  // and more accurate.
  double r1 = cuda_random_double(rand_state);
  double r2 = cuda_random_double(rand_state);
  double z = 1 + r2 * (sqrt(1 - radius * radius / dist2) - 1);
  double phi = 2 * CUDA_PI * r1;
  double x = cos(phi) * sqrt(1 - z * z);
  double y = sin(phi) * sqrt(1 - z * z);

  return uvw.transform(CudaVec3(x, y, z));
}

#endif // USE_CUDA