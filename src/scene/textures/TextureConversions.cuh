#pragma once

#ifdef USE_CUDA

#include <stdexcept>

#include "../../utils/math/PerlinNoiseConversions.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "CheckerTexture.hpp"
#include "NoiseTexture.hpp"
#include "SolidColorTexture.hpp"
#include "Texture.cuh"
#include "Texture.hpp"
#include "TextureTypes.hpp"

// Forward declaration of main conversion function.
inline CudaTexture cpu_to_cuda_texture(const Texture &cpu_texture);

// Convert CPU SolidColorTexture to CUDA Texture.
inline CudaTexture
cpu_to_cuda_solid_color_texture(const SolidColorTexture &solid_texture) {
  CudaTexture cuda_texture;
  cuda_texture.type = CudaTextureType::TEXTURE_SOLID;

  cuda_texture.solid =
      new CudaSolidColorTexture(cpu_to_cuda_vec3(solid_texture.get_albedo()));

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

  cuda_texture.checker =
      new CudaCheckerTexture(checker_texture.get_scale(), &even_tex, &odd_tex);

  return cuda_texture;
}

// Convert CPU NoiseTexture to CUDA Texture.
inline CudaTexture
cpu_to_cuda_noise_texture(const NoiseTexture &noise_texture) {
  CudaTexture cuda_texture;
  cuda_texture.type = CudaTextureType::TEXTURE_NOISE;

  CudaPerlinNoise cuda_perlin =
      cpu_to_cuda_perlin_noise(noise_texture.get_perlin());
  cuda_texture.noise =
      new CudaNoiseTexture(noise_texture.get_scale(), cuda_perlin);

  return cuda_texture;
}

// Generic texture conversion with runtime type detection.
inline CudaTexture cpu_to_cuda_texture(const Texture &cpu_texture) {
  // Use polymorphism to detect texture type by attempting casts.
  if (const SolidColorTexture *solid_texture =
          dynamic_cast<const SolidColorTexture *>(&cpu_texture)) {
    return cpu_to_cuda_solid_color_texture(*solid_texture);
  } else if (const CheckerTexture *checker_texture =
                 dynamic_cast<const CheckerTexture *>(&cpu_texture)) {
    return cpu_to_cuda_checker_texture(*checker_texture);
  } else if (const NoiseTexture *noise_texture =
                 dynamic_cast<const NoiseTexture *>(&cpu_texture)) {
    return cpu_to_cuda_noise_texture(*noise_texture);
  } else {
    throw std::runtime_error(
        "TextureConversions.cuh::cpu_to_cuda_texture: Unknown texture type "
        "encountered during CPU to CUDA texture conversion. Unable to convert "
        "unrecognized texture object.");
  }
}

// Convert CUDA Texture to CPU - creates appropriate CPU texture based on type.
inline TexturePtr cuda_to_cpu_texture(const CudaTexture &cuda_texture) {
  switch (cuda_texture.type) {
  case CudaTextureType::TEXTURE_SOLID: {
    Color cpu_color = cuda_to_cpu_vec3(cuda_texture.solid->albedo);
    return std::make_shared<SolidColorTexture>(cpu_color);
  }
  case CudaTextureType::TEXTURE_CHECKER: {
    // Extract colors from embedded checker texture data.
    const CudaCheckerTexture &checker_data = *cuda_texture.checker;

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
    PerlinNoise cpu_perlin_noise =
        cuda_to_cpu_perlin_noise(cuda_texture.noise->perlin);
    return std::make_shared<NoiseTexture>(cuda_texture.noise->scale,
                                          cpu_perlin_noise);
  }
  default:
    throw std::runtime_error(
        "TextureConversions.cuh::cuda_to_cpu_texture: Unknown CUDA texture "
        "type encountered during CUDA to CPU texture conversion. Invalid "
        "texture type in switch statement.");
  }
}

#endif // USE_CUDA