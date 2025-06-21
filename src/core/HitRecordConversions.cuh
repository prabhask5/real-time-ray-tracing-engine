#pragma once

#ifdef USE_CUDA

#include "../scene/materials/MaterialConversions.cuh"
#include "../utils/math/Vec3Conversions.cuh"
#include "HitRecord.cuh"
#include "HitRecord.hpp"

// Convert CPU HitRecord to CUDA HitRecord.
inline CudaHitRecord cpu_to_cuda_hit_record(const HitRecord &cpu_record) {
  CudaHitRecord cuda_record;

  cuda_record.point = cpu_to_cuda_vec3(cpu_record.point);
  cuda_record.normal = cpu_to_cuda_vec3(cpu_record.normal);
  cuda_record.t = cpu_record.t;
  cuda_record.front_face = cpu_record.frontFace;
  cuda_record.u = cpu_record.u;
  cuda_record.v = cpu_record.v;

  // Convert material with proper memory management.
  if (cpu_record.material_pointer != nullptr) {
    CudaMaterial cuda_material = cpu_to_cuda_material(*cpu_record.material);
    cuda_record.material_pointer = &cuda_material;
  } else {
    cuda_record.material_pointer = nullptr;
  }

  return cuda_record;
}

// Convert CUDA HitRecord to CPU HitRecord.
inline HitRecord cuda_to_cpu_hit_record(const CudaHitRecord &cuda_record) {
  HitRecord cpu_record;

  cpu_record.point = cuda_to_cpu_vec3(cuda_record.point);
  cpu_record.normal = cuda_to_cpu_vec3(cuda_record.normal);
  cpu_record.t = cuda_record.t;
  cpu_record.frontFace = cuda_record.front_face;
  cpu_record.u = cuda_record.u;
  cpu_record.v = cuda_record.v;

  // Convert material with proper reconstruction from CUDA data.
  if (cuda_record.material_pointer != nullptr)
    cpu_record.material = cuda_to_cpu_material(*cuda_record.material_pointer);
  else
    cpu_record.material = nullptr;

  return cpu_record;
}

#endif // USE_CUDA