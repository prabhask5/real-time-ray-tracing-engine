#ifdef USE_CUDA

#include "../core/Hittable.cuh"
#include "../utils/memory/CudaMemoryUtility.cuh"
#include "BVHNode.cuh"
#include <iomanip>
#include <sstream>

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

// JSON serialization function for CudaBVHNode.
std::string cuda_json_bvh_node(const CudaBVHNode &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaBVHNode\",";
  oss << "\"address\":\"" << &obj << "\",";
  if (obj.left) {
    CudaHittable host_left;
    cudaMemcpyDeviceToHostSafe(&host_left, obj.left, 1);
    oss << "\"left\":" << cuda_json_hittable(host_left) << ",";
  } else {
    oss << "\"left\":null,";
  }
  if (obj.right) {
    CudaHittable host_right;
    cudaMemcpyDeviceToHostSafe(&host_right, obj.right, 1);
    oss << "\"right\":" << cuda_json_hittable(host_right) << ",";
  } else {
    oss << "\"right\":null,";
  }
  oss << "\"bbox\":" << cuda_json_aabb(obj.bbox) << ",";
  oss << "\"is_leaf\":" << (obj.is_leaf ? "true" : "false");
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA