#pragma once

#ifdef USE_CUDA

#include "../../utils/math/PerlinNoiseConversions.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "CheckerTexture.hpp"
#include "NoiseTexture.hpp"
#include "SolidColorTexture.hpp"
#include "Texture.cuh"
#include "Texture.hpp"
#include "TextureTypes.hpp"

// Forward declarations.
class SolidColorTexture;
class CheckerTexture;
class NoiseTexture;

// Convert CPU SolidColorTexture to CUDA Texture.
inline CudaTexture
cpu_to_cuda_solid_color_texture(const SolidColorTexture &solid_texture) {
  CudaTexture cuda_texture;
  cuda_texture.type = CudaTextureType::TEXTURE_SOLID;

  cuda_texture.data.solid =
      CudaSolidColorTexture(cpu_to_cuda_vec3(solid_texture.get_albedo()));

  return cuda_texture;
}

// Convert CPU CheckerTexture to CUDA Texture.
inline CudaTexture
cpu_to_cuda_checker_texture(const CheckerTexture &checker_texture) {
  CudaTexture cuda_texture;
  cuda_texture.type = CudaTextureType::TEXTURE_CHECKER;

  // Create embedded textures for the checker pattern.
  CudaTexture even_tex =
      cpu_to_cuda_texture(*checker_texture.get_even_texture());
  CudaTexture odd_tex = cpu_to_cuda_texture(*checker_texture.get_odd_texture());

  cuda_texture.data.checker =
      CudaCheckerTexture(checker_texture.get_scale(), &even_tex, &odd_tex,
                         even_tex.type, odd_tex.type);

  return cuda_texture;
}

// Convert CPU NoiseTexture to CUDA Texture.
inline CudaTexture
cpu_to_cuda_noise_texture(const NoiseTexture &noise_texture) {
  CudaTexture cuda_texture;
  cuda_texture.type = CudaTextureType::TEXTURE_NOISE;

  CudaPerlinNoise cuda_perlin =
      cpu_to_cuda_perlin_noise(noise_texture.get_perlin());
  cuda_texture.data.noise =
      CudaNoiseTexture(noise_texture.get_scale(), cuda_perlin);

  return cuda_texture;
}

// Generic texture conversion with runtime type detection.
inline CudaTexture cpu_to_cuda_texture(const Texture &cpu_texture) {
  // Use polymorphism to detect texture type by attempting casts.
  if (auto solid_texture =
          dynamic_cast<const SolidColorTexture *>(&cpu_texture)) {
    return cpu_to_cuda_solid_color_texture(*solid_texture);
  } else if (auto checker_texture =
                 dynamic_cast<const CheckerTexture *>(&cpu_texture)) {
    return cpu_to_cuda_checker_texture(*checker_texture);
  } else if (auto noise_texture =
                 dynamic_cast<const NoiseTexture *>(&cpu_texture)) {
    return cpu_to_cuda_noise_texture(*noise_texture);
  } else {
    // Fallback: create solid color texture by sampling.
    Color sampled_color = cpu_texture.value(0.5, 0.5, Point3(0, 0, 0));
    return cuda_make_solid_texture(sampled_color);
  }
}

// Convert CUDA Texture to CPU - creates appropriate CPU texture based on type.
inline TexturePtr cuda_to_cpu_texture(const CudaTexture &cuda_texture) {
  switch (cuda_texture.type) {
  case CudaTextureType::TEXTURE_SOLID: {
    Color cpu_color = cuda_to_cpu_vec3(cuda_texture.data.solid.albedo);
    return std::make_shared<SolidColorTexture>(cpu_color);
  }
  case CudaTextureType::TEXTURE_CHECKER: {
    // Extract colors from embedded checker texture data.
    const auto &checker_data = cuda_texture.data.checker;

    // Convert embedded textures back to CPU.
    Color even_color, odd_color;
    if (checker_data.even_texture != nullptr &&
        checker_data.odd_texture != nullptr) {
      TexturePtr even_cpu_texture =
          cuda_to_cpu_texture(*checker_data.even_texture);
      TexturePtr odd_cpu_texture =
          cuda_to_cpu_texture(*checker_data.odd_texture);

      return std::make_shared<CheckerTexture>(
          checker_data.scale, even_cpu_texture, odd_cpu_texture);
    } else {
      // Default colors if embedded textures are null.
      even_color = Color(0.2, 0.3, 0.1);
      odd_color = Color(0.9, 0.9, 0.9);
      return std::make_shared<CheckerTexture>(checker_data.scale, even_color,
                                              odd_color);
    }
  }
  case CudaTextureType::TEXTURE_NOISE: {
    return std::make_shared<NoiseTexture>(cuda_texture.data.noise.scale,
                                          cuda_texture.data.noise.perlin);
  }
  default:
    return std::make_shared<SolidColorTexture>(Color(0.7, 0.7, 0.7));
  }
}

// Batch conversion functions.
void batch_cpu_to_cuda_texture(const Texture **cpu_textures,
                               CudaTexture *cuda_textures, int count);
void batch_cuda_to_cpu_texture(const CudaTexture *cuda_textures,
                               TexturePtr *cpu_textures, int count);

#endif // USE_CUDA