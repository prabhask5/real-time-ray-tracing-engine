#ifdef USE_CUDA

#include "../core/Hittable.cuh"
#include "BVHNode.cuh"

CudaBVHNode cuda_make_bvh_node(CudaHittable *left, CudaHittable *right,
                               bool is_leaf) {
  CudaBVHNode node;
  node.left = left;
  node.right = right;
  node.is_leaf = is_leaf;

  // Compute bounding box from children.
  CudaAABB left_box = cuda_hittable_get_bounding_box(*left);
  CudaAABB right_box = cuda_hittable_get_bounding_box(*right);
  node.bbox = cuda_make_aabb(left_box, right_box);
  return node;
}

__device__ bool cuda_bvh_node_hit(const CudaBVHNode &node, const CudaRay &ray,
                                  CudaInterval t_values, CudaHitRecord &record,
                                  curandState *rand_state) {
  // Early exit if ray doesn't hit bounding box.
  if (!cuda_aabb_hit(node.bbox, ray, t_values))
    return false;

  CudaHitRecord temp_record;
  bool hit_left =
      cuda_hittable_hit(*node.left, ray, t_values, temp_record, rand_state);

  if (hit_left) {
    t_values.max = temp_record.t;
    record = temp_record;
  }

  bool hit_right =
      cuda_hittable_hit(*node.right, ray, t_values, temp_record, rand_state);
  if (hit_right && temp_record.t < record.t)
    record = temp_record;

  return hit_left || hit_right;
}

__device__ double cuda_bvh_node_pdf_value(const CudaBVHNode &node,
                                          const CudaPoint3 &origin,
                                          const CudaVec3 &direction) {
  return 0.5 * cuda_hittable_pdf_value(*node.left, origin, direction) +
         0.5 * cuda_hittable_pdf_value(*node.right, origin, direction);
}

__device__ CudaVec3 cuda_bvh_node_random(const CudaBVHNode &node,
                                         const CudaPoint3 &origin,
                                         curandState *state) {
  if (cuda_random_double(state) < 0.5)
    return cuda_hittable_random(*node.left, origin, state);
  return cuda_hittable_random(*node.right, origin, state);
}

#endif // USE_CUDA