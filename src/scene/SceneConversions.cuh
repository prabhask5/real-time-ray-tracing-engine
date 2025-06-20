#pragma once

#ifdef USE_CUDA

#include "../core/HittableList.hpp"
#include "../core/HittableListConversions.cuh"
#include "../utils/math/Vec3Conversions.cuh"
#include "materials/MaterialConversions.cuh"
#include "objects/Plane.hpp"
#include "objects/PlaneConversions.cuh"
#include "objects/Sphere.hpp"
#include "objects/SphereConversions.cuh"
#include "textures/TextureConversions.cuh"

// Comprehensive scene conversion structure.
struct CudaSceneData {
  CudaHittableList world;
  CudaHittableList lights;
  CudaHittable *world_objects_buffer;
  CudaHittable *lights_objects_buffer;
  int world_object_count;
  int lights_object_count;
};

// Convert entire scene from CPU to CUDA with proper material handling.
inline CudaSceneData
convert_complete_scene_to_cuda(const HittableList &cpu_world,
                               const HittableList &cpu_lights) {
  CudaSceneData scene_data;

  // Allocate buffers for objects.
  const int max_objects = 64;
  scene_data.world_objects_buffer = new CudaHittable[max_objects];
  scene_data.lights_objects_buffer = new CudaHittable[max_objects];

  // Convert world objects.
  const auto &world_objects = cpu_world.get_objects();
  scene_data.world_object_count =
      std::min((int)world_objects.size(), max_objects);

  for (int i = 0; i < scene_data.world_object_count; i++) {
    const auto &cpu_object = world_objects[i];

    if (auto sphere = std::dynamic_pointer_cast<Sphere>(cpu_object)) {
      // Use complete sphere conversion function
      scene_data.world_objects_buffer[i].type = HITTABLE_SPHERE;
      scene_data.world_objects_buffer[i].sphere = cpu_to_cuda_sphere(*sphere);
    } else if (auto plane = std::dynamic_pointer_cast<Plane>(cpu_object)) {
      // Use complete plane conversion function
      scene_data.world_objects_buffer[i].type = HITTABLE_PLANE;
      scene_data.world_objects_buffer[i].plane = cpu_to_cuda_plane(*plane);
    } else {
      // Default object
      CudaTexture default_texture =
          cuda_make_solid_texture(Color(0.5, 0.5, 0.5));
      CudaMaterial default_material =
          cuda_make_lambertian_material(default_texture);
      scene_data.world_objects_buffer[i].type = HITTABLE_SPHERE;
      scene_data.world_objects_buffer[i].sphere =
          create_cuda_sphere_static(Point3(0, 0, 0), 1.0, default_material);
    }
  }

  // Convert light objects
  const auto &light_objects = cpu_lights.get_objects();
  scene_data.lights_object_count =
      std::min((int)light_objects.size(), max_objects);

  for (int i = 0; i < scene_data.lights_object_count; i++) {
    const auto &cpu_object = light_objects[i];

    if (auto sphere = std::dynamic_pointer_cast<Sphere>(cpu_object)) {
      // Use complete sphere conversion function
      scene_data.lights_objects_buffer[i].type = HITTABLE_SPHERE;
      scene_data.lights_objects_buffer[i].sphere = cpu_to_cuda_sphere(*sphere);
    } else if (auto plane = std::dynamic_pointer_cast<Plane>(cpu_object)) {
      // Use complete plane conversion function
      scene_data.lights_objects_buffer[i].type = HITTABLE_PLANE;
      scene_data.lights_objects_buffer[i].plane = cpu_to_cuda_plane(*plane);
    } else {
      CudaTexture default_texture =
          cuda_make_solid_texture(Color(1.0, 1.0, 1.0));
      CudaMaterial default_material =
          cuda_make_lambertian_material(default_texture);
      scene_data.lights_objects_buffer[i].type = HITTABLE_SPHERE;
      scene_data.lights_objects_buffer[i].sphere =
          create_cuda_sphere_static(Point3(0, 0, 0), 1.0, default_material);
    }
  }

  // Create CUDA hittable lists using the buffers
  scene_data.world = create_cuda_hittable_list_with_buffer(
      scene_data.world_objects_buffer, scene_data.world_object_count);
  scene_data.lights = create_cuda_hittable_list_with_buffer(
      scene_data.lights_objects_buffer, scene_data.lights_object_count);

  return scene_data;
}

// Cleanup scene data
inline void cleanup_cuda_scene_data(CudaSceneData &scene_data) {
  delete[] scene_data.world_objects_buffer;
  delete[] scene_data.lights_objects_buffer;
  scene_data.world_objects_buffer = nullptr;
  scene_data.lights_objects_buffer = nullptr;
}

// Convert scene with specific Cornell Box setup
inline CudaSceneData create_cornell_box_cuda_scene() {
  CudaSceneData scene_data;
  scene_data.world_objects_buffer =
      new CudaHittable[10]; // Enough for Cornell Box
  scene_data.lights_objects_buffer = new CudaHittable[5];

  // Create materials
  CudaTexture red_texture = cuda_make_solid_texture(Color(0.65, 0.05, 0.05));
  CudaMaterial red_material = cuda_make_lambertian_material(red_texture);
  CudaTexture white_texture = cuda_make_solid_texture(Color(0.73, 0.73, 0.73));
  CudaMaterial white_material = cuda_make_lambertian_material(white_texture);
  CudaTexture green_texture = cuda_make_solid_texture(Color(0.12, 0.45, 0.15));
  CudaMaterial green_material = cuda_make_lambertian_material(green_texture);
  CudaTexture light_texture = cuda_make_solid_texture(Color(15, 15, 15));
  CudaMaterial light_material = cuda_make_diffuse_light_material(light_texture);
  CudaMaterial glass_material = cuda_make_dielectric_material(1.5);

  // Create Cornell Box walls
  int obj_count = 0;

  // Left wall (green)
  scene_data.world_objects_buffer[obj_count].type = HITTABLE_PLANE;
  scene_data.world_objects_buffer[obj_count].plane = create_cuda_plane(
      Point3(555, 0, 0), Vec3(0, 0, 555), Vec3(0, 555, 0), green_material);
  obj_count++;

  // Right wall (red)
  scene_data.world_objects_buffer[obj_count].type = HITTABLE_PLANE;
  scene_data.world_objects_buffer[obj_count].plane = create_cuda_plane(
      Point3(0, 0, 555), Vec3(0, 0, -555), Vec3(0, 555, 0), red_material);
  obj_count++;

  // Floor (white)
  scene_data.world_objects_buffer[obj_count].type = HITTABLE_PLANE;
  scene_data.world_objects_buffer[obj_count].plane = create_cuda_plane(
      Point3(0, 555, 0), Vec3(555, 0, 0), Vec3(0, 0, 555), white_material);
  obj_count++;

  // Back wall (white)
  scene_data.world_objects_buffer[obj_count].type = HITTABLE_PLANE;
  scene_data.world_objects_buffer[obj_count].plane = create_cuda_plane(
      Point3(0, 0, 555), Vec3(555, 0, 0), Vec3(0, 0, -555), white_material);
  obj_count++;

  // Ceiling (white)
  scene_data.world_objects_buffer[obj_count].type = HITTABLE_PLANE;
  scene_data.world_objects_buffer[obj_count].plane = create_cuda_plane(
      Point3(555, 0, 555), Vec3(-555, 0, 0), Vec3(0, 555, 0), white_material);
  obj_count++;

  // Light
  scene_data.world_objects_buffer[obj_count].type = HITTABLE_PLANE;
  scene_data.world_objects_buffer[obj_count].plane = create_cuda_plane(
      Point3(213, 554, 227), Vec3(130, 0, 0), Vec3(0, 0, 105), light_material);
  obj_count++;

  // Glass sphere
  scene_data.world_objects_buffer[obj_count].type = HITTABLE_SPHERE;
  scene_data.world_objects_buffer[obj_count].sphere =
      create_cuda_sphere_static(Point3(190, 90, 190), 90, glass_material);
  obj_count++;

  scene_data.world_object_count = obj_count;

  // Create lights list
  int light_count = 0;

  // Light plane for sampling
  scene_data.lights_objects_buffer[light_count].type = HITTABLE_PLANE;
  scene_data.lights_objects_buffer[light_count].plane =
      create_cuda_plane(Point3(343, 554, 332), Vec3(-130, 0, 0),
                        Vec3(0, 0, -105), white_material);
  light_count++;

  // Glass sphere for light sampling
  scene_data.lights_objects_buffer[light_count].type = HITTABLE_SPHERE;
  scene_data.lights_objects_buffer[light_count].sphere =
      create_cuda_sphere_static(Point3(190, 90, 190), 90, white_material);
  light_count++;

  scene_data.lights_object_count = light_count;

  // Create hittable lists using the buffers
  scene_data.world = create_cuda_hittable_list_with_buffer(
      scene_data.world_objects_buffer, scene_data.world_object_count);
  scene_data.lights = create_cuda_hittable_list_with_buffer(
      scene_data.lights_objects_buffer, scene_data.lights_object_count);

  return scene_data;
}

#endif // USE_CUDA