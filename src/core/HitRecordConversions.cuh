#pragma once

#ifdef USE_CUDA

#include "../scene/materials/Material.cuh"
#include "../scene/materials/MaterialConversions.cuh"
#include "../utils/math/Vec3Conversions.cuh"
#include "HitRecord.cuh"
#include "HitRecord.hpp"

// Convert CPU HitRecord to CUDA HitRecord POD struct.
inline CudaHitRecord cpu_to_cuda_hit_record(const HitRecord &cpu_record) {
  CudaHitRecord cuda_record;
  cuda_record.point = cpu_to_cuda_vec3(cpu_record.point);
  cuda_record.normal = cpu_to_cuda_vec3(cpu_record.normal);
  cuda_record.t = cpu_record.t;
  cuda_record.front_face = cpu_record.frontFace;
  cuda_record.u = cpu_record.u;
  cuda_record.v = cpu_record.v;

  // Leave material alone for now.
  cuda_record.material = nullptr;

  return cuda_record;
}

#endif // USE_CUDA