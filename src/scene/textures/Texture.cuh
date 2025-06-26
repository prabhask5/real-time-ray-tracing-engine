#pragma once

#ifdef USE_CUDA

#include "../../core/Vec3Types.cuh"
#include "../../utils/math/PerlinNoise.cuh"
#include "../../utils/math/Vec3.cuh"

enum class CudaTextureType { TEXTURE_SOLID, TEXTURE_CHECKER, TEXTURE_NOISE };

// Forward declarations.
struct CudaTexture;

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
  const CudaTexture *even_texture;
  const CudaTexture *odd_texture;
};

// Checker texture initialization function.
__device__ inline CudaCheckerTexture
cuda_make_checker_texture(double scale, const CudaTexture *even_texture,
                          const CudaTexture *odd_texture) {
  CudaCheckerTexture texture;
  texture.scale = scale;
  texture.even_texture = even_texture;
  texture.odd_texture = odd_texture;
  return texture;
}

// Checker texture value function.
__device__ CudaColor cuda_checker_texture_value(
    const CudaCheckerTexture &texture, double u, double v, const CudaPoint3 &p);

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
    CudaSolidColorTexture *solid;
    CudaCheckerTexture *checker;
    CudaNoiseTexture *noise;
  };
};

// Texture value function.
__device__ CudaColor cuda_texture_value(const CudaTexture &texture, double u,
                                        double v, const CudaPoint3 &p);

// Helper texture constructor functions.
__device__ CudaTexture cuda_make_texture_solid(CudaColor albedo);
__device__ CudaTexture
cuda_make_texture_checker(double scale, const CudaTexture *even_texture,
                          const CudaTexture *odd_texture);
__device__ CudaTexture cuda_make_texture_noise(double scale,
                                               CudaPerlinNoise perlin);

#endif // USE_CUDA
