#pragma once

#ifdef USE_CUDA

#include "../core/HittableConversions.cuh"
#include <cuda_runtime.h>

// Comprehensive scene conversion structure.
struct CudaSceneData {
  CudaHittable world;
  CudaHittable lights;
  CudaHittable *world_objects_buffer;
  CudaHittable *lights_objects_buffer;
};

// Convert entire scene from CPU to CUDA with proper material handling.
inline CudaSceneData
convert_complete_scene_to_cuda(const Hittable &cpu_world,
                               const Hittable &cpu_lights) {
  CudaSceneData scene_data;

  // Convert world objects.
  CudaHittable raw_world;

  try {
    raw_world = cpu_to_cuda_hittable(cpu_world);
  } catch (const std::exception &e) {
    std::cerr << "Error converting world object: " << e.what() << std::endl;
    scene_data.world_objects_buffer = nullptr;
    scene_data.lights_objects_buffer = nullptr;
    return scene_data;
  }

  // Convert light objects.
  CudaHittable raw_lights;

  try {
    raw_lights = cpu_to_cuda_hittable(cpu_lights);
  } catch (const std::exception &e) {
    std::cerr << "Error converting lights object: " << e.what() << std::endl;
    scene_data.world_objects_buffer = nullptr;
    scene_data.lights_objects_buffer = nullptr;
    return scene_data;
  }

  // Allocate device memory for the buffers.
  size_t world_buffer_size =
      raw_world.hittable_list->count * sizeof(CudaHittable);
  size_t lights_buffer_size =
      raw_lights.hittable_list->count * sizeof(CudaHittable);

  std::clog << "Allocating CUDA memory: world=" << world_buffer_size
            << " bytes, lights=" << lights_buffer_size << " bytes\n";

  cudaError_t world_alloc =
      cudaMalloc(&scene_data.world_objects_buffer, world_buffer_size);

  if (world_alloc != cudaSuccess) {
    std::cerr << "Failed to allocate CUDA memory for world objects: "
              << cudaGetErrorString(world_alloc) << std::endl;
    scene_data.world_objects_buffer = nullptr;
    scene_data.lights_objects_buffer = nullptr;
    return scene_data;
  }

  // Handle empty lights list case.
  if (lights_buffer_size > 0) {
    cudaError_t lights_alloc =
        cudaMalloc(&scene_data.lights_objects_buffer, lights_buffer_size);

    if (lights_alloc != cudaSuccess) {
      std::cerr << "Failed to allocate CUDA memory for light objects: "
                << cudaGetErrorString(lights_alloc) << std::endl;
      cudaFree(scene_data.world_objects_buffer);
      scene_data.world_objects_buffer = nullptr;
      scene_data.lights_objects_buffer = nullptr;
      return scene_data;
    }
  } else {
    std::clog << "No lights to allocate, setting lights buffer to nullptr\n";
    scene_data.lights_objects_buffer = nullptr;
  }

  // Copy host buffers to device memory.
  cudaMemcpy(scene_data.world_objects_buffer,
             raw_world.hittable_list->hittables, world_buffer_size,
             cudaMemcpyHostToDevice);

  if (lights_buffer_size > 0) {
    cudaMemcpy(scene_data.lights_objects_buffer,
               raw_lights.hittable_list->hittables, lights_buffer_size,
               cudaMemcpyHostToDevice);
  }

  // Create CUDA hittable lists using the device buffers.

  raw_world.hittable_list->hittables = scene_data.world_objects_buffer;
  scene_data.world.type = raw_world.type;
  scene_data.world.hittable_list = raw_world.hittable_list;

  raw_lights.hittable_list->hittables = scene_data.lights_objects_buffer;
  scene_data.lights.type = raw_lights.type;
  scene_data.lights.hittable_list = raw_lights.hittable_list;

  return scene_data;
}

// Cleanup scene data.
inline void cleanup_cuda_scene_data(CudaSceneData &scene_data) {
  if (scene_data.world_objects_buffer) {
    cudaFree(scene_data.world_objects_buffer);
    scene_data.world_objects_buffer = nullptr;
  }
  if (scene_data.lights_objects_buffer) {
    cudaFree(scene_data.lights_objects_buffer);
    scene_data.lights_objects_buffer = nullptr;
  }
}

#endif // USE_CUDA