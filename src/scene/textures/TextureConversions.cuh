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
  CudaColor cuda_albedo = cpu_to_cuda_vec3(solid_texture.get_albedo());
  return cuda_make_texture_solid(cuda_albedo);
}

// Convert CPU CheckerTexture to CUDA Texture.
inline CudaTexture
cpu_to_cuda_checker_texture(const CheckerTexture &checker_texture) {
  CudaTexture *even_tex = new CudaTexture();
  CudaTexture *odd_tex = new CudaTexture();

  *even_tex = cpu_to_cuda_texture(*checker_texture.get_even_texture());
  *odd_tex = cpu_to_cuda_texture(*checker_texture.get_odd_texture());

  return cuda_make_texture_checker(checker_texture.get_scale(), even_tex,
                                   odd_tex);
}

// Convert CPU NoiseTexture to CUDA Texture.
inline CudaTexture
cpu_to_cuda_noise_texture(const NoiseTexture &noise_texture) {
  CudaPerlinNoise cuda_perlin =
      cpu_to_cuda_perlin_noise(noise_texture.get_perlin());
  return cuda_make_texture_noise(noise_texture.get_scale(), cuda_perlin);
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
    throw std::runtime_error(
        "TextureConversions.cuh::cpu_to_cuda_texture: Unknown texture type "
        "encountered during CPU to CUDA texture conversion. Unable to convert "
        "unrecognized texture object.");
  }
}

#endif // USE_CUDA