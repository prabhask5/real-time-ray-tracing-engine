#pragma once

#ifdef USE_CUDA

#include "../../utils/memory/CudaSceneContext.cuh"
#include "../textures/TextureConverter.cuh"
#include "DielectricMaterial.hpp"
#include "DiffuseLightMaterial.hpp"
#include "IsotropicMaterial.hpp"
#include "LambertianMaterial.hpp"
#include "Material.cuh"
#include "Material.hpp"
#include "MaterialTypes.hpp"
#include "MetalMaterial.hpp"
#include <climits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

// Material collection and indexing system.
class MaterialConverter {
public:
  MaterialConverter(CudaSceneContext &ctx, TextureConverter &tex_converter)
      : m_context(ctx), m_texture_converter(tex_converter) {}

  size_t get_material_index(MaterialPtr material) {
    if (material == nullptr) {
      return static_cast<size_t>(-1); // Return SIZE_MAX for null materials.
    }

    const Material *raw_ptr = material.get();

    auto it = m_material_to_index.find(raw_ptr);
    if (it != m_material_to_index.end()) {
      return it->second;
    }

    // Convert material to CUDA format and add to context.
    CudaMaterial cuda_material = cpu_to_cuda_material(material);
    size_t index = m_context.add_material(cuda_material);
    m_material_to_index[raw_ptr] = index;
    return index;
  }

private:
  std::unordered_map<const Material *, size_t> m_material_to_index;
  CudaSceneContext &m_context;
  TextureConverter &m_texture_converter;

private:
  // Convert CPU LambertianMaterial to CUDA Material.
  CudaMaterial
  cpu_to_cuda_lambertian_material(const LambertianMaterial &lambertian) {
    // Get texture index through texture converter.
    size_t texture_index =
        m_texture_converter.get_texture_index(lambertian.get_texture());

    return cuda_make_material_lambertian(texture_index);
  }

  // Convert CPU MetalMaterial to CUDA Material.
  CudaMaterial cpu_to_cuda_metal_material(const MetalMaterial &metal) {
    Color cpu_albedo = metal.get_albedo();
    double fuzz = metal.get_fuzz();
    CudaColor cuda_albedo = cpu_to_cuda_vec3(cpu_albedo);

    return cuda_make_material_metal(cuda_albedo, fuzz);
  }

  // Convert CPU DielectricMaterial to CUDA Material.
  CudaMaterial
  cpu_to_cuda_dielectric_material(const DielectricMaterial &dielectric) {
    return cuda_make_material_dielectric(dielectric.get_refraction_index());
  }

  // Convert CPU DiffuseLightMaterial to CUDA Material.
  CudaMaterial
  cpu_to_cuda_diffuse_light_material(const DiffuseLightMaterial &light) {
    // Get texture index through texture converter.
    size_t texture_index =
        m_texture_converter.get_texture_index(light.get_texture());

    return cuda_make_material_diffuse_light(texture_index);
  }

  // Convert CPU IsotropicMaterial to CUDA Material.
  CudaMaterial
  cpu_to_cuda_isotropic_material(const IsotropicMaterial &isotropic) {
    // Get texture index through texture converter.
    size_t texture_index =
        m_texture_converter.get_texture_index(isotropic.get_texture());

    return cuda_make_material_isotropic(texture_index);
  }

  // Generic material conversion with runtime type detection.
  CudaMaterial cpu_to_cuda_material(MaterialPtr cpu_material) {
    // Try to cast to different material types.
    if (auto lambertian =
            dynamic_cast<const LambertianMaterial *>(cpu_material.get())) {
      return cpu_to_cuda_lambertian_material(*lambertian);
    } else if (auto metal =
                   dynamic_cast<const MetalMaterial *>(cpu_material.get())) {
      return cpu_to_cuda_metal_material(*metal);
    } else if (auto dielectric = dynamic_cast<const DielectricMaterial *>(
                   cpu_material.get())) {
      return cpu_to_cuda_dielectric_material(*dielectric);
    } else if (auto light = dynamic_cast<const DiffuseLightMaterial *>(
                   cpu_material.get())) {
      return cpu_to_cuda_diffuse_light_material(*light);
    } else if (auto isotropic = dynamic_cast<const IsotropicMaterial *>(
                   cpu_material.get())) {
      return cpu_to_cuda_isotropic_material(*isotropic);
    } else {
      throw std::runtime_error("MaterialConversions.cuh::cpu_to_cuda_material: "
                               "Unknown material type "
                               "encountered during CPU to CUDA material "
                               "conversion. Unable to convert "
                               "unrecognized material object.");
    }
  }
};

#endif // USE_CUDA