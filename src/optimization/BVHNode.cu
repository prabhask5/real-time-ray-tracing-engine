#ifdef USE_CUDA

#include "../core/Hittable.cuh"
#include "BVHNode.cuh"

__device__ CudaBVHNode::CudaBVHNode(CudaHittable *_left, CudaHittable *_right,
                                    bool _is_leaf)
    : left(_left), right(_right), is_leaf(_is_leaf) {
  // Compute bounding box from children.
  CudaAABB left_box = left->get_bounding_box();
  CudaAABB right_box = right->get_bounding_box();
  bbox = CudaAABB(left_box, right_box);
}

__device__ bool CudaBVHNode::hit(const CudaRay &ray, CudaInterval t_values,
                                 CudaHitRecord &record,
                                 curandState *rand_state) const {
  // Early exit if ray doesn't hit bounding box.
  if (!bbox.hit(ray, t_values))
    return false;

  CudaHitRecord temp_record;
  bool hit_left = left->hit(ray, t_values, temp_record, rand_state);

  if (hit_left) {
    t_values.max = temp_record.t;
    record = temp_record;
  }

  bool hit_right = right->hit(ray, t_values, temp_record, rand_state);
  if (hit_right && temp_record.t < record.t)
    record = temp_record;

  return hit_left || hit_right;
}

__device__ double CudaBVHNode::pdf_value(const CudaPoint3 &origin,
                                         const CudaVec3 &direction) const {
  return 0.5 * left->pdf_value(origin, direction) +
         0.5 * right->pdf_value(origin, direction);
}

__device__ CudaVec3 CudaBVHNode::random(const CudaPoint3 &origin,
                                        curandState *state) const {
  if (cuda_random_double(state) < 0.5)
    return left->random(origin, state);
  return right->random(origin, state);
}

#endif // USE_CUDA