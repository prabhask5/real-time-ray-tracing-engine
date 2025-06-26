#ifdef USE_CUDA

#include "Plane.cuh"

CudaPlane cuda_make_plane(const CudaPoint3 &corner, const CudaVec3 &u_side,
                          const CudaVec3 &v_side, size_t material_index) {
  CudaPlane plane;
  plane.corner = corner;
  plane.u_side = u_side;
  plane.v_side = v_side;
  plane.material_index = material_index;

  CudaVec3 n = cuda_vec3_cross_product(u_side, v_side);
  plane.normal = cuda_vec3_unit_vector(n);
  plane.D = cuda_vec3_dot_product(plane.normal, corner);
  plane.w = cuda_vec3_divide_scalar(n, cuda_vec3_dot_product(n, n));
  plane.surface_area = cuda_vec3_length(n);

  CudaPoint3 p0 = corner;
  CudaPoint3 p1 = cuda_vec3_add(corner, u_side);
  CudaPoint3 p2 = cuda_vec3_add(corner, v_side);
  CudaPoint3 p3 = cuda_vec3_add(cuda_vec3_add(corner, u_side), v_side);

  CudaAABB box1 = cuda_make_aabb(p0, p3);
  CudaAABB box2 = cuda_make_aabb(p1, p2);
  plane.bbox = cuda_make_aabb(box1, box2);
  return plane;
}

__device__ bool cuda_plane_hit(const CudaPlane &plane, const CudaRay &ray,
                               CudaInterval t_values, CudaHitRecord &record,
                               curandState *rand_state) {
  double denom = cuda_vec3_dot_product(plane.normal, ray.direction);

  if (fabs(denom) < 1e-8)
    return false;

  double t =
      (plane.D - cuda_vec3_dot_product(plane.normal, ray.origin)) / denom;
  if (!cuda_interval_contains(t_values, t))
    return false;

  CudaPoint3 intersection = cuda_ray_at(ray, t);
  CudaVec3 planar_hit = cuda_vec3_subtract(intersection, plane.corner);
  double alpha = cuda_vec3_dot_product(
      plane.w, cuda_vec3_cross_product(planar_hit, plane.v_side));
  double beta = cuda_vec3_dot_product(
      plane.w, cuda_vec3_cross_product(plane.u_side, planar_hit));

  if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1)
    return false;
  record.u = alpha;
  record.v = beta;

  record.t = t;
  record.point = intersection;
  cuda_hit_record_set_face_normal(record, ray, plane.normal);
  record.material_index = plane.material_index;

  return true;
}

__device__ double cuda_plane_pdf_value(const CudaPlane &plane,
                                       const CudaPoint3 &origin,
                                       const CudaVec3 &direction) {
  CudaRay ray = cuda_make_ray(origin, direction);
  CudaHitRecord record;
  if (!cuda_plane_hit(plane, ray, cuda_make_interval(0.001, CUDA_INF), record,
                      nullptr))
    return 0.0;

  double distance_squared =
      record.t * record.t * cuda_vec3_length_squared(direction);
  double cosine = fabs(cuda_vec3_dot_product(direction, record.normal) /
                       cuda_vec3_length(direction));

  return distance_squared / (cosine * plane.surface_area);
}

__device__ CudaVec3 cuda_plane_random(const CudaPlane &plane,
                                      const CudaPoint3 &origin,
                                      curandState *rand_state) {
  double u = cuda_random_double(rand_state);
  double v = cuda_random_double(rand_state);

  CudaPoint3 point_on_plane = cuda_vec3_add(
      cuda_vec3_add(plane.corner, cuda_vec3_multiply_scalar(plane.u_side, u)),
      cuda_vec3_multiply_scalar(plane.v_side, v));
  return cuda_vec3_subtract(point_on_plane, origin);
}

#endif // USE_CUDA