#pragma once

#ifdef USE_CUDA

#include "../../core/Vec3Types.cuh"
#include "../../utils/math/PerlinNoise.cuh"
#include "../../utils/math/Vec3.cuh"
#include "../../utils/memory/CudaSceneContext.cuh"
#include <iomanip>
#include <sstream>

enum class CudaTextureType { TEXTURE_SOLID, TEXTURE_CHECKER, TEXTURE_NOISE };

// Forward declarations.
struct CudaTexture;

// Forward declaration of global device functions.
__device__ const CudaTexture &cuda_get_texture(size_t index);

// POD struct for solid color texture.
struct CudaSolidColorTexture {
  CudaColor albedo;
};

// Solid color texture initialization function.
__device__ inline CudaSolidColorTexture
cuda_make_solid_color_texture(CudaColor albedo) {
  CudaSolidColorTexture texture;
  texture.albedo = albedo;
  return texture;
}

// Solid color texture value function.
__device__ inline CudaColor
cuda_solid_color_texture_value(const CudaSolidColorTexture &texture, double u,
                               double v, const CudaPoint3 &p) {
  return texture.albedo;
}

// POD struct for checker texture.
struct CudaCheckerTexture {
  double scale;
  size_t even_texture_index;
  size_t odd_texture_index;
};

// Checker texture initialization function.
__device__ inline CudaCheckerTexture
cuda_make_checker_texture(double scale, size_t even_texture_index,
                          size_t odd_texture_index) {
  CudaCheckerTexture texture;
  texture.scale = scale;
  texture.even_texture_index = even_texture_index;
  texture.odd_texture_index = odd_texture_index;
  return texture;
}

// Checker texture value function.
__device__ inline CudaColor
cuda_checker_texture_value(const CudaCheckerTexture &texture, double u,
                           double v, const CudaPoint3 &p);

// POD struct for noise texture.
struct CudaNoiseTexture {
  double scale;
  CudaPerlinNoise perlin;
};

// Noise texture initialization function.
__device__ inline CudaNoiseTexture
cuda_make_noise_texture(double scale, CudaPerlinNoise perlin) {
  CudaNoiseTexture texture;
  texture.scale = scale;
  texture.perlin = perlin;
  return texture;
}

// Noise texture value function.
__device__ inline CudaColor
cuda_noise_texture_value(const CudaNoiseTexture &texture, double u, double v,
                         const CudaPoint3 &p) {
  CudaColor base_color = cuda_make_vec3(0.5, 0.5, 0.5);
  double noise_value = 1.0 + sin(texture.scale * p.z +
                                 10.0 * cuda_perlin_turb(texture.perlin, p, 7));
  return cuda_vec3_multiply_scalar(base_color, noise_value);
}

// POD struct for unified texture using manual dispatch pattern.
struct CudaTexture {
  CudaTextureType type;
  union {
    CudaSolidColorTexture solid;
    CudaCheckerTexture checker;
    CudaNoiseTexture noise;
  };
};

// Texture value function.
__device__ inline CudaColor cuda_texture_value(const CudaTexture &texture,
                                               double u, double v,
                                               const CudaPoint3 &p) {
  switch (texture.type) {
  case CudaTextureType::TEXTURE_SOLID:
    return cuda_solid_color_texture_value(texture.solid, u, v, p);
  case CudaTextureType::TEXTURE_CHECKER:
    return cuda_checker_texture_value(texture.checker, u, v, p);
  case CudaTextureType::TEXTURE_NOISE:
    return cuda_noise_texture_value(texture.noise, u, v, p);
  default:
    // ERROR: Texture.cu::cuda_texture_value - Unknown Texture type in switch
    // statement. This should never happen in well-formed code.
    return cuda_make_vec3(0.0, 0.0, 0.0); // Safe fallback for GPU device code.
  }
}

// Helper texture constructor functions.
__device__ inline CudaTexture cuda_make_texture_solid_color(CudaColor albedo) {
  CudaTexture texture;
  texture.type = CudaTextureType::TEXTURE_SOLID;
  texture.solid = cuda_make_solid_color_texture(albedo);
  return texture;
}

__device__ inline CudaTexture
cuda_make_texture_checker(double scale, size_t even_texture_index,
                          size_t odd_texture_index) {
  CudaTexture texture;
  texture.type = CudaTextureType::TEXTURE_CHECKER;
  texture.checker =
      cuda_make_checker_texture(scale, even_texture_index, odd_texture_index);
  return texture;
}

__device__ inline CudaTexture cuda_make_texture_noise(double scale,
                                                      CudaPerlinNoise perlin) {
  CudaTexture texture;
  texture.type = CudaTextureType::TEXTURE_NOISE;
  texture.noise = cuda_make_noise_texture(scale, perlin);
  return texture;
}

// Checker texture implementation now that CudaTexture is defined.
__device__ inline CudaColor
cuda_checker_texture_value(const CudaCheckerTexture &texture, double u,
                           double v, const CudaPoint3 &p) {
  double inv_scale = 1.0 / texture.scale;

  int x_index = int(floor(inv_scale * p.x));
  int y_index = int(floor(inv_scale * p.y));
  int z_index = int(floor(inv_scale * p.z));

  bool is_even = (x_index + y_index + z_index) % 2 == 0;

  return is_even
             ? cuda_texture_value(cuda_get_texture(texture.even_texture_index),
                                  u, v, p)
             : cuda_texture_value(cuda_get_texture(texture.odd_texture_index),
                                  u, v, p);
}

// JSON serialization functions for CUDA textures.
inline std::string
cuda_json_solid_color_texture(const CudaSolidColorTexture &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaSolidColorTexture\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"albedo\":" << cuda_json_vec3(obj.albedo);
  oss << "}";
  return oss.str();
}

inline std::string cuda_json_checker_texture(const CudaCheckerTexture &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaCheckerTexture\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"scale\":" << obj.scale << ",";
  oss << "\"even_texture_index\":" << obj.even_texture_index << ",";
  oss << "\"odd_texture_index\":" << obj.odd_texture_index;
  oss << "}";
  return oss.str();
}

inline std::string cuda_json_noise_texture(const CudaNoiseTexture &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaNoiseTexture\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"scale\":" << obj.scale << ",";
  oss << "\"perlin\":" << cuda_json_perlin_noise(obj.perlin);
  oss << "}";
  return oss.str();
}

inline std::string cuda_json_texture(const CudaTexture &obj) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{";
  oss << "\"type\":\"CudaTexture\",";
  oss << "\"address\":\"" << &obj << "\",";
  oss << "\"texture_type\":";
  switch (obj.type) {
  case CudaTextureType::TEXTURE_SOLID:
    oss << "\"SOLID\",";
    oss << "\"solid\":" << cuda_json_solid_color_texture(obj.solid);
    break;
  case CudaTextureType::TEXTURE_CHECKER:
    oss << "\"CHECKER\",";
    oss << "\"checker\":" << cuda_json_checker_texture(obj.checker);
    break;
  case CudaTextureType::TEXTURE_NOISE:
    oss << "\"NOISE\",";
    oss << "\"noise\":" << cuda_json_noise_texture(obj.noise);
    break;
  }
  oss << "}";
  return oss.str();
}

#endif // USE_CUDA