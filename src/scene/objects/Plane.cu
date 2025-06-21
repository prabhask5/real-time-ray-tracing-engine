#ifdef USE_CUDA

#include "Plane.cuh"

__device__
CudaPlane::CudaPlane(const CudaPoint3 &_corner, const CudaVec3 &_u_side,
                     const CudaVec3 &_v_side, const CudaMaterial *_material)
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

__device__ bool CudaPlane::hit(const CudaRay &ray, CudaInterval t_values,
                               CudaHitRecord &record,
                               curandState *rand_state) const {

  double denom = cuda_dot_product(normal, ray.direction);

  // No hit if the ray is parallel to the plane.
  if (fabs(denom) < 1e-8)
    return false;

  // Return false if the hit point parameter t is outside the ray interval.
  double t = (D - cuda_dot_product(normal, ray.origin)) / denom;
  if (!t_values.contains(t))
    return false;

  // Determine if the hit point lies within the planar shape using its plane
  // coordinates.
  CudaPoint3 intersection = ray.at(t);
  CudaVec3 planar_hit = intersection - corner;
  double alpha = cuda_dot_product(w, cuda_cross_product(planar_hit, v_side));
  double beta = cuda_dot_product(w, cuda_cross_product(u_side, planar_hit));

  // Given the hit point in plane coordinates, return false if it is outside
  // the primitive, otherwise set the hit record UV coordinates and return
  // true.
  if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1)
    return false;
  record.u = alpha;
  record.v = beta;

  // Ray hits the 2D shape; set the rest of the hit record and return true.
  record.t = t;
  record.point = intersection;
  cuda_set_face_normal(record, ray, normal);
  record.material_pointer = const_cast<CudaMaterial *>(material);

  return true;
}

__device__ double CudaPlane::pdf_value(const CudaPoint3 &origin,
                                       const CudaVec3 &direction) const {

  CudaRay ray = CudaRay(origin, direction);
  CudaHitRecord record;
  if (!this->hit(ray, CudaInterval(0.001, CUDA_INF), record, nullptr))
    return 0.0;

  double distance_squared = record.t * record.t * direction.length_squared();
  double cosine =
      fabs(cuda_dot_product(direction, record.normal) / direction.length());

  // PDF formula for solid angle sampling.
  return distance_squared / (cosine * surface_area);
}

__device__ CudaVec3 CudaPlane::random(const CudaPoint3 &origin,
                                      curandState *rand_state) const {

  double u = cuda_random_double(rand_state);
  double v = cuda_random_double(rand_state);

  // Pick a random point on the rectangle.
  CudaPoint3 point_on_plane = corner + u * u_side + v * v_side;
  return point_on_plane - origin;
}

#endif // USE_CUDA