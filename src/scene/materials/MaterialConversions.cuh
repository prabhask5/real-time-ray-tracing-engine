#pragma once

#ifdef USE_CUDA

#include <stdexcept>

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
  TexturePtr cpu_texture = lambertian.get_texture();
  CudaTexture *cuda_texture = new CudaTexture();
  *cuda_texture = cpu_to_cuda_texture(*cpu_texture);

  return cuda_make_material_lambertian(cuda_texture);
}

// Convert CPU MetalMaterial to CUDA Material.
inline CudaMaterial cpu_to_cuda_metal_material(const MetalMaterial &metal) {
  Color cpu_albedo = metal.get_albedo();
  double fuzz = metal.get_fuzz();
  CudaColor cuda_albedo = cpu_to_cuda_vec3(cpu_albedo);

  return cuda_make_material_metal(cuda_albedo, fuzz);
}

// Convert CPU DielectricMaterial to CUDA Material.
inline CudaMaterial
cpu_to_cuda_dielectric_material(const DielectricMaterial &dielectric) {
  return cuda_make_material_dielectric(dielectric.get_refraction_index());
}

// Convert CPU DiffuseLightMaterial to CUDA Material.
inline CudaMaterial
cpu_to_cuda_diffuse_light_material(const DiffuseLightMaterial &light) {
  TexturePtr cpu_texture = light.get_texture();
  CudaTexture *cuda_texture = new CudaTexture();
  *cuda_texture = cpu_to_cuda_texture(*cpu_texture);

  return cuda_make_material_diffuse_light(cuda_texture);
}

// Convert CPU IsotropicMaterial to CUDA Material.
inline CudaMaterial
cpu_to_cuda_isotropic_material(const IsotropicMaterial &isotropic) {
  TexturePtr cpu_texture = isotropic.get_texture();
  CudaTexture *cuda_texture = new CudaTexture();
  *cuda_texture = cpu_to_cuda_texture(*cpu_texture);

  return cuda_make_material_isotropic(cuda_texture);
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
    throw std::runtime_error(
        "MaterialConversions.cuh::cpu_to_cuda_material: Unknown material type "
        "encountered during CPU to CUDA material conversion. Unable to convert "
        "unrecognized material object.");
  }
}

#endif // USE_CUDA