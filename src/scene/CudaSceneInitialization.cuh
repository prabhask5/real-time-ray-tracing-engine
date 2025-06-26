#pragma once

#ifdef USE_CUDA

#include "../core/Hittable.hpp"
#include "../core/HittableConverter.cuh"
#include "../core/HittableList.hpp"
#include "../optimization/BVHNode.hpp"
#include "../utils/memory/CudaMemoryUtility.cuh"
#include "../utils/memory/CudaSceneContext.cuh"
#include "../utils/memory/CudaUniquePtr.cuh"
#include "materials/DiffuseLightMaterial.hpp"
#include "materials/IsotropicMaterial.hpp"
#include "materials/LambertianMaterial.hpp"
#include "materials/MaterialConverter.cuh"
#include "mediums/ConstantMedium.hpp"
#include "objects/Plane.hpp"
#include "objects/RotateY.hpp"
#include "objects/Sphere.hpp"
#include "objects/Translate.hpp"
#include "textures/CheckerTexture.hpp"
#include "textures/TextureConverter.cuh"
#include <memory>
#include <unordered_set>

// Structure to hold scene complexity analysis results.
struct SceneComplexityStats {
  // Object counts.
  size_t sphere_count = 0;
  size_t plane_count = 0;
  size_t bvh_node_count = 0;
  size_t rotate_y_count = 0;
  size_t translate_count = 0;
  size_t constant_medium_count = 0;
  size_t hittable_list_count = 0;
  size_t total_list_objects = 0; // Sum of all objects in all lists.

  // Resource counts (unique).
  size_t material_count = 0;
  size_t texture_count = 0;

  // Memory size estimates (in bytes).
  size_t estimated_object_memory = 0;
  size_t estimated_total_memory = 0;

  void calculate_memory_estimates() {
    // Calculate object memory requirements.
    estimated_object_memory =
        sphere_count * sizeof(CudaSphere) + plane_count * sizeof(CudaPlane) +
        bvh_node_count * sizeof(CudaBVHNode) +
        rotate_y_count * sizeof(CudaRotateY) +
        translate_count * sizeof(CudaTranslate) +
        constant_medium_count * sizeof(CudaConstantMedium) +
        hittable_list_count * sizeof(CudaHittableList) +
        total_list_objects * sizeof(CudaHittable); // Objects within lists.

    // Add material and texture memory.
    size_t material_memory = material_count * sizeof(CudaMaterial);
    size_t texture_memory = texture_count * sizeof(CudaTexture);

    // Add 50% overhead for alignment, fragmentation, and safety margin.
    estimated_total_memory =
        (estimated_object_memory + material_memory + texture_memory) * 3 / 2;

    // Ensure minimum 16MB buffer.
    if (estimated_total_memory < 16 * 1024 * 1024) {
      estimated_total_memory = 16 * 1024 * 1024;
    }
  }
};

// Structure to hold all CUDA scene data.
struct CudaSceneData {
  CudaUniquePtr<CudaHittable> world;
  CudaUniquePtr<CudaHittable> lights;
};

// Forward declaration for recursive analysis.
inline void
analyze_hittable_object(const Hittable &hittable, SceneComplexityStats &stats,
                        std::unordered_set<const Material *> &unique_materials,
                        std::unordered_set<const Texture *> &unique_textures);

// Analyze materials and textures to count unique instances.
inline void analyze_material_resources(
    MaterialPtr material,
    std::unordered_set<const Material *> &unique_materials,
    std::unordered_set<const Texture *> &unique_textures) {
  if (!material)
    return;

  const Material *material_ptr = material.get();
  if (unique_materials.find(material_ptr) != unique_materials.end()) {
    return;
  }
  unique_materials.insert(material_ptr);

  // Analyze material-specific textures.
  if (auto lambertian =
          dynamic_cast<const LambertianMaterial *>(material_ptr)) {
    TexturePtr texture = lambertian->get_texture();
    if (texture)
      unique_textures.insert(texture.get());
  } else if (auto diffuse_light =
                 dynamic_cast<const DiffuseLightMaterial *>(material_ptr)) {
    TexturePtr texture = diffuse_light->get_texture();
    if (texture)
      unique_textures.insert(texture.get());
  } else if (auto isotropic =
                 dynamic_cast<const IsotropicMaterial *>(material_ptr)) {
    TexturePtr texture = isotropic->get_texture();
    if (texture)
      unique_textures.insert(texture.get());
  }

  // Recursive texture analysis for complex textures like CheckerTexture.
  for (const Texture *tex_ptr : unique_textures) {
    if (auto checker = dynamic_cast<const CheckerTexture *>(tex_ptr)) {
      TexturePtr even_tex = checker->get_even_texture();
      TexturePtr odd_tex = checker->get_odd_texture();
      if (even_tex)
        unique_textures.insert(even_tex.get());
      if (odd_tex)
        unique_textures.insert(odd_tex.get());
    }
  }
}

// Recursively analyze hittable objects and count types.
inline void
analyze_hittable_object(const Hittable &hittable, SceneComplexityStats &stats,
                        std::unordered_set<const Material *> &unique_materials,
                        std::unordered_set<const Texture *> &unique_textures) {
  if (auto sphere = dynamic_cast<const Sphere *>(&hittable)) {
    stats.sphere_count++;
    analyze_material_resources(sphere->get_material(), unique_materials,
                               unique_textures);
  } else if (auto plane = dynamic_cast<const Plane *>(&hittable)) {
    stats.plane_count++;
    analyze_material_resources(plane->get_material(), unique_materials,
                               unique_textures);
  } else if (auto bvh_node = dynamic_cast<const BVHNode *>(&hittable)) {
    stats.bvh_node_count++;

    // Recursively analyze left and right children.
    analyze_hittable_object(*bvh_node->get_left(), stats, unique_materials,
                            unique_textures);
    analyze_hittable_object(*bvh_node->get_right(), stats, unique_materials,
                            unique_textures);
  } else if (auto rotate_y = dynamic_cast<const RotateY *>(&hittable)) {
    stats.rotate_y_count++;

    // Recursively analyze wrapped object.
    analyze_hittable_object(*rotate_y->get_object(), stats, unique_materials,
                            unique_textures);
  } else if (auto translate = dynamic_cast<const Translate *>(&hittable)) {
    stats.translate_count++;

    // Recursively analyze wrapped object.
    analyze_hittable_object(*translate->get_object(), stats, unique_materials,
                            unique_textures);
  } else if (auto constant_medium =
                 dynamic_cast<const ConstantMedium *>(&hittable)) {
    stats.constant_medium_count++;

    // Analyze boundary object and phase function material.
    analyze_hittable_object(*constant_medium->get_boundary(), stats,
                            unique_materials, unique_textures);
    analyze_material_resources(constant_medium->get_phase_function(),
                               unique_materials, unique_textures);
  } else if (auto hittable_list =
                 dynamic_cast<const HittableList *>(&hittable)) {
    stats.hittable_list_count++;
    const std::vector<HittablePtr> &objects = hittable_list->get_objects();
    stats.total_list_objects += objects.size();

    // Recursively analyze all objects in the list.
    for (const HittablePtr &obj : objects) {
      analyze_hittable_object(*obj, stats, unique_materials, unique_textures);
    }
  }
}

// Comprehensive scene analysis function.
inline SceneComplexityStats
analyze_scene_complexity(const Hittable &cpu_world,
                         const Hittable &cpu_lights) {
  SceneComplexityStats stats;
  std::unordered_set<const Material *> unique_materials;
  std::unordered_set<const Texture *> unique_textures;

  // Analyze world objects.
  analyze_hittable_object(cpu_world, stats, unique_materials, unique_textures);

  // Analyze light objects.
  analyze_hittable_object(cpu_lights, stats, unique_materials, unique_textures);

  // Update counts from unique sets.
  stats.material_count = unique_materials.size();
  stats.texture_count = unique_textures.size();

  // Calculate memory estimates.
  stats.calculate_memory_estimates();

  return stats;
}

// Main initialization function.
inline CudaSceneData initialize_cuda_scene(const Hittable &cpu_world,
                                           const Hittable &cpu_lights) {
  std::clog << "DEBUG: step 1: Getting singleton context" << std::endl;
  // Get singleton scene context.
  CudaSceneContext &context = CudaSceneContext::get_context();
  std::clog << "DEBUG: step 2: Got context successfully" << std::endl;

  std::clog << "DEBUG: step 3: Starting scene analysis" << std::endl;
  // Perform comprehensive scene analysis.
  SceneComplexityStats complexity_stats =
      analyze_scene_complexity(cpu_world, cpu_lights);
  std::clog << "DEBUG: step 4: Scene analysis complete" << std::endl;

  std::clog << "DEBUG: step 5: Calculating buffer size" << std::endl;
  // Calculate buffer size in MB (add extra safety margin).
  size_t buffer_size_mb = std::max(
      16UL, (complexity_stats.estimated_total_memory + 1024 * 1024 - 1) /
                (1024 * 1024));
  std::clog << "DEBUG: step 6: Buffer size: " << buffer_size_mb << " MB"
            << std::endl;

  std::clog << "DEBUG: step 7: Initializing context" << std::endl;
  // Initialize scene context with accurate estimates.
  context.initialize(complexity_stats.material_count,
                     complexity_stats.texture_count, buffer_size_mb);
  std::clog << "DEBUG: step 8: Context initialized" << std::endl;

  std::clog << "DEBUG: step 9: Creating converters" << std::endl;
  // Create material and texture converters.
  TextureConverter texture_converter(context);
  MaterialConverter material_converter(context, texture_converter);
  HittableConverter hittable_converter(material_converter, texture_converter,
                                       context);
  std::clog << "DEBUG: step 10: Converters created" << std::endl;

  std::clog << "DEBUG: step 11: Creating shared pointers" << std::endl;
  // Convert all objects to CUDA format.
  HittablePtr world_shared =
      HittablePtr(const_cast<Hittable *>(&cpu_world), [](Hittable *) {});
  HittablePtr lights_shared =
      HittablePtr(const_cast<Hittable *>(&cpu_lights), [](Hittable *) {});
  std::clog << "DEBUG: step 12: Shared pointers created" << std::endl;

  std::clog << "DEBUG: step 13: Converting world hittables" << std::endl;
  CudaHittable cuda_world =
      hittable_converter.cpu_to_cuda_hittable(world_shared);
  std::clog << "DEBUG: step 14: World converted" << std::endl;

  std::clog << "DEBUG: step 15: Converting light hittables" << std::endl;
  CudaHittable cuda_lights =
      hittable_converter.cpu_to_cuda_hittable(lights_shared);
  std::clog << "DEBUG: step 16: Lights converted" << std::endl;

  std::clog << "DEBUG: step 17: Finalizing context" << std::endl;
  // Finalize and upload resource arrays to GPU.
  context.finalize_and_upload();
  std::clog << "DEBUG: step 18: Context finalized" << std::endl;

  std::clog << "DEBUG: step 19: Initializing device context" << std::endl;
  // Initialize device context pointer for kernel access.
  initialize_device_scene_context();
  std::clog << "DEBUG: step 20: Device context initialized" << std::endl;

  std::clog << "DEBUG: step 21: Creating scene data struct" << std::endl;
  // Allocate and copy world and light objects to device.
  CudaSceneData scene_data;
  std::clog << "DEBUG: step 22: Copying world to device" << std::endl;
  scene_data.world = make_cuda_unique_from_host(&cuda_world, 1);
  std::clog << "DEBUG: step 23: World copied to device" << std::endl;
  scene_data.lights = make_cuda_unique_from_host(&cuda_lights, 1);
  std::clog << "DEBUG: step 24: Lights copied to device" << std::endl;
  std::clog << "DEBUG: step 25: Returning scene data" << std::endl;
  return scene_data;
}

// Cleanup CUDA scene data.
inline void cleanup_cuda_scene(CudaSceneData &scene_data) {
  // CudaUniquePtr will automatically clean up when destructed.
  scene_data.world.reset();
  scene_data.lights.reset();

  // Clean up device context.
  cleanup_device_scene_context();

  // Clean up scene context.
  CudaSceneContext::destroy_context();
}

#endif // USE_CUDA