#pragma once

#ifdef USE_CUDA

#include "../../utils/math/PerlinNoiseConversions.cuh"
#include "../../utils/math/Vec3Conversions.cuh"
#include "../../utils/memory/CudaSceneContext.cuh"
#include "CheckerTexture.hpp"
#include "NoiseTexture.hpp"
#include "SolidColorTexture.hpp"
#include "Texture.cuh"
#include "Texture.hpp"
#include "TextureTypes.hpp"
#include <memory>
#include <stdexcept>
#include <unordered_map>

// Texture collection and indexing system.
class TextureConverter {
public:
  TextureConverter(CudaSceneContext &ctx) : m_context(ctx) {}

  size_t get_texture_index(TexturePtr texture) {
    const Texture *raw_ptr = texture.get();

    auto it = m_texture_to_index.find(raw_ptr);
    if (it != m_texture_to_index.end()) {
      return it->second;
    }

    // Convert texture to CUDA format and add to context.
    CudaTexture cuda_texture = cpu_to_cuda_texture(texture);
    size_t index = m_context.add_texture(cuda_texture);
    m_texture_to_index[raw_ptr] = index;
    return index;
  }

private:
  std::unordered_map<const Texture *, size_t> m_texture_to_index;
  CudaSceneContext &m_context;

private:
  // Convert CPU SolidColorTexture to CUDA Texture.
  CudaTexture
  cpu_to_cuda_solid_color_texture(const SolidColorTexture &solid_texture) {
    CudaColor cuda_albedo = cpu_to_cuda_vec3(solid_texture.get_albedo());
    return cuda_make_texture_solid_color(cuda_albedo);
  }

  // Convert CPU CheckerTexture to CUDA Texture.
  CudaTexture
  cpu_to_cuda_checker_texture(const CheckerTexture &checker_texture) {
    size_t even_tex_index =
        get_texture_index(checker_texture.get_even_texture());
    size_t odd_tex_index = get_texture_index(checker_texture.get_odd_texture());

    return cuda_make_texture_checker(checker_texture.get_scale(),
                                     even_tex_index, odd_tex_index);
  }

  // Convert CPU NoiseTexture to CUDA Texture.
  CudaTexture cpu_to_cuda_noise_texture(const NoiseTexture &noise_texture) {
    CudaPerlinNoise cuda_perlin =
        cpu_to_cuda_perlin_noise(noise_texture.get_perlin());
    return cuda_make_texture_noise(noise_texture.get_scale(), cuda_perlin);
  }

  // Generic texture conversion with runtime type detection.
  CudaTexture cpu_to_cuda_texture(TexturePtr cpu_texture) {
    // Use polymorphism to detect texture type by attempting casts.
    if (auto solid_texture =
            dynamic_cast<const SolidColorTexture *>(cpu_texture.get())) {
      return cpu_to_cuda_solid_color_texture(*solid_texture);
    } else if (auto checker_texture =
                   dynamic_cast<const CheckerTexture *>(cpu_texture.get())) {
      return cpu_to_cuda_checker_texture(*checker_texture);
    } else if (auto noise_texture =
                   dynamic_cast<const NoiseTexture *>(cpu_texture.get())) {
      return cpu_to_cuda_noise_texture(*noise_texture);
    } else {
      throw std::runtime_error(
          "TextureConversions.cuh::cpu_to_cuda_texture: Unknown texture type "
          "encountered during CPU to CUDA texture conversion. Unable to "
          "convert "
          "unrecognized texture object.");
    }
  }
};

#endif // USE_CUDA