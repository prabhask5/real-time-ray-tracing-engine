#ifdef USE_CUDA

#include "../../core/Hittable.cuh"
#include "ConstantMedium.cuh"

__device__ CudaConstantMedium::CudaConstantMedium(const CudaHittable *_boundary,
                                                  double density,
                                                  CudaTexture *texture)
    : boundary(_boundary), neg_inv_density(-1.0 / density),
      phase_function(new CudaMaterial(cuda_make_isotropic_material(texture))) {}

__device__ bool CudaConstantMedium::hit(const CudaRay &ray,
                                        CudaInterval t_values,
                                        CudaHitRecord &record,
                                        curandState *rand_state) const {

  CudaHitRecord rec1, rec2;

  // Find if the ray hits and enters the boundary.
  if (!boundary->hit(ray, CudaInterval(-CUDA_INF, CUDA_INF), rec1, rand_state))
    return false;

  // Find if the ray exits the boundary, only if the ray enters the boundary.
  if (!boundary->hit(ray, CudaInterval(rec1.t + 0.0001, CUDA_INF), rec2,
                     rand_state))
    return false;

  if (rec1.t < t_values.min)
    rec1.t = t_values.min;
  if (rec2.t > t_values.max)
    rec2.t = t_values.max;

  if (rec1.t >= rec2.t)
    return false;

  if (rec1.t < 0)
    rec1.t = 0;

  // This simulates a random scattering distance inside the medium using an
  // exponential distribution (common for modeling scattering).
  double ray_length = ray.direction.length();
  double distance_inside = (rec2.t - rec1.t) * ray_length;

  // This randomly determines when the ray scatters inside the medium.
  double hit_distance = neg_inv_density * log(cuda_random_double(rand_state));

  // If the scattering distance is longer than the ray distance inside the
  // medium boundary, it does not scatter.
  if (hit_distance > distance_inside)
    return false;

  record.t = rec1.t + hit_distance / ray_length;
  record.point = ray.at(record.t);
  record.normal = CudaVec3(1, 0, 0); // arbitrary
  record.front_face = true;
  record.material_pointer = const_cast<CudaMaterial *>(phase_function);

  return true;
}

__device__ CudaAABB CudaConstantMedium::get_bounding_box() const {
  return boundary->get_bounding_box();
}

#endif // USE_CUDA