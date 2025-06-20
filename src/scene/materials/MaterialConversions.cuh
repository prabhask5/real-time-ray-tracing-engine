#pragma once

#ifdef USE_CUDA

#include "../../utils/math/Vec3Conversions.cuh"
#include "../textures/TextureConversions.cuh"
#include "DielectricMaterial.hpp"
#include "DiffuseLightMaterial.hpp"
#include "IsotropicMaterial.hpp"
#include "LambertianMaterial.hpp"
#include "Material.cuh"
#include "Material.hpp"
#include "MaterialTypes.hpp"
#include "MetalMaterial.hpp"

// Convert CPU LambertianMaterial to CUDA Material.
inline CudaMaterial
cpu_to_cuda_lambertian_material(const LambertianMaterial &lambertian) {
  CudaMaterial cuda_material;
  cuda_material.type = CudaMaterialType::MATERIAL_LAMBERTIAN;

  // Extract texture and convert it.
  TexturePtr cpu_texture = lambertian.get_texture();
  CudaTexture cuda_texture = cpu_to_cuda_texture(*cpu_texture);

  cuda_material.data.lambertian = CudaLambertianMaterial(cuda_texture);
  return cuda_material;
}

// Convert CPU MetalMaterial to CUDA Material.
inline CudaMaterial cpu_to_cuda_metal_material(const MetalMaterial &metal) {
  CudaMaterial cuda_material;
  cuda_material.type = CudaMaterialType::MATERIAL_METAL;

  // Extract albedo and fuzz using getter methods.
  Color cpu_albedo = metal.get_albedo();
  double fuzz = metal.get_fuzz();
  CudaColor cuda_albedo = cpu_to_cuda_vec3(cpu_albedo);

  cuda_material.data.metal = CudaMetalMaterial(cuda_albedo, fuzz);
  return cuda_material;
}

// Convert CPU DielectricMaterial to CUDA Material.
inline CudaMaterial
cpu_to_cuda_dielectric_material(const DielectricMaterial &dielectric) {
  CudaMaterial cuda_material;
  cuda_material.type = CudaMaterialType::MATERIAL_DIELECTRIC;

  // Extract refraction index.
  cuda_material.data.dielectric =
      CudaDielectricMaterial(dielectric.get_refraction_index());
  return cuda_material;
}

// Convert CPU DiffuseLightMaterial to CUDA Material.
inline CudaMaterial
cpu_to_cuda_diffuse_light_material(const DiffuseLightMaterial &light) {
  CudaMaterial cuda_material;
  cuda_material.type = CudaMaterialType::MATERIAL_DIFFUSE_LIGHT;

  // Extract texture and convert it.
  TexturePtr cpu_texture = light.get_texture();
  CudaTexture cuda_texture = cpu_to_cuda_texture(*cpu_texture);

  cuda_material.data.diffuse = CudaDiffuseLightMaterial(cuda_texture);
  return cuda_material;
}

// Convert CPU IsotropicMaterial to CUDA Material.
inline CudaMaterial
cpu_to_cuda_isotropic_material(const IsotropicMaterial &isotropic) {
  CudaMaterial cuda_material;
  cuda_material.type = CudaMaterialType::MATERIAL_ISOTROPIC;

  // Extract texture and convert it.
  TexturePtr cpu_texture = isotropic.get_texture();
  CudaTexture cuda_texture = cpu_to_cuda_texture(*cpu_texture);

  cuda_material.data.isotropic = CudaIsotropicMaterial(cuda_texture);
  return cuda_material;
}

// Generic material conversion with runtime type detection.
inline CudaMaterial cpu_to_cuda_material(const Material &cpu_material) {
  // Try to cast to different material types.
  if (auto lambertian =
          dynamic_cast<const LambertianMaterial *>(&cpu_material)) {
    return cpu_to_cuda_lambertian_material(*lambertian);
  } else if (auto metal = dynamic_cast<const MetalMaterial *>(&cpu_material)) {
    return cpu_to_cuda_metal_material(*metal);
  } else if (auto dielectric =
                 dynamic_cast<const DielectricMaterial *>(&cpu_material)) {
    return cpu_to_cuda_dielectric_material(*dielectric);
  } else if (auto light =
                 dynamic_cast<const DiffuseLightMaterial *>(&cpu_material)) {
    return cpu_to_cuda_diffuse_light_material(*light);
  } else if (auto isotropic =
                 dynamic_cast<const IsotropicMaterial *>(&cpu_material)) {
    return cpu_to_cuda_isotropic_material(*isotropic);
  } else {
    // Default to Lambertian.
    CudaMaterial cuda_material;
    cuda_material.type = CudaMaterialType::MATERIAL_LAMBERTIAN;
    CudaTexture default_texture = cuda_make_solid_texture(Color(0.7, 0.7, 0.7));
    cuda_material.data.lambertian = CudaLambertianMaterial(default_texture);
    return cuda_material;
  }
}

// Convert CUDA Material to CPU Material.
inline MaterialPtr cuda_to_cpu_material(const CudaMaterial &cuda_material) {
  switch (cuda_material.type) {
  case CudaMaterialType::MATERIAL_LAMBERTIAN: {
    // Extract texture and create Lambertian material.
    auto cpu_texture =
        cuda_to_cpu_texture(cuda_material.data.lambertian.texture);
    return std::make_shared<LambertianMaterial>(cpu_texture);
  }
  case CudaMaterialType::MATERIAL_METAL: {
    Color albedo = cuda_to_cpu_vec3(cuda_material.data.metal.albedo);
    return std::make_shared<MetalMaterial>(albedo,
                                           cuda_material.data.metal.fuzz);
  }
  case CudaMaterialType::MATERIAL_DIELECTRIC: {
    return std::make_shared<DielectricMaterial>(
        cuda_material.data.dielectric.refraction_index);
  }
  case CudaMaterialType::MATERIAL_DIFFUSE_LIGHT: {
    auto cpu_texture = cuda_to_cpu_texture(cuda_material.data.diffuse.texture);
    return std::make_shared<DiffuseLightMaterial>(cpu_texture);
  }
  case CudaMaterialType::MATERIAL_ISOTROPIC: {
    auto cpu_texture =
        cuda_to_cpu_texture(cuda_material.data.isotropic.texture);
    return std::make_shared<IsotropicMaterial>(cpu_texture);
  }
  default:
    return std::make_shared<LambertianMaterial>(Color(0.7, 0.7, 0.7));
  }
}

// Batch conversion functions.
void batch_cpu_to_cuda_material(const Material **cpu_materials,
                                CudaMaterial *cuda_materials, int count);
void batch_cuda_to_cpu_material(const CudaMaterial *cuda_materials,
                                MaterialPtr *cpu_materials, int count);

#endif // USE_CUDA