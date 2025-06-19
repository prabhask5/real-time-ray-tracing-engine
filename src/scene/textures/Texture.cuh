#pragma once

#ifdef USE_CUDA

#include "../../core/Vec3Types.cuh"
#include "../../utils/math/PerlinNoise.cuh"
#include "../../utils/math/Vec3.cuh"

enum class CudaTextureType { TEXTURE_SOLID, TEXTURE_CHECKER, TEXTURE_NOISE };

struct CudaSolidColorTexture {
  CudaColor albedo;

  __device__ CudaSolidColorTexture(CudaColor _albedo) : albedo(_albedo) {}
};

struct CudaCheckerTexture {
  double scale;
  const void *even_texture;
  const void *odd_texture;
  CudaTextureType even_type;
  CudaTextureType odd_type;

  __device__ CudaCheckerTexture(double _scale, const void *_even_texture,
                                const void *_odd_texture,
                                CudaTextureType _even_type,
                                CudaTextureType _odd_type)
      : scale(_scale), even_texture(_even_texture), odd_texture(_odd_texture),
        even_type(_even_type), odd_type(_odd_type) {}
};

struct CudaNoiseTexture {
  double scale;
  CudaPerlinNoise perlin;

  __device__ CudaNoiseTexture(double _scale, CudaPerlinNoise _perlin)
      : scale(_scale), perlin(_perlin) {}
};

struct CudaTexture {
  CudaTextureType type;
  union {
    CudaSolidColorTexture solid;
    CudaCheckerTexture checker;
    CudaNoiseTexture noise;
  } data;

  __device__ inline CudaColor value(double u, double v, const CudaPoint3 &p) {
    switch (type) {
    case CudaTextureType::TEXTURE_SOLID:
      return data.solid.albedo;

    case CudaTextureType::TEXTURE_NOISE: {
      double t = data.noise.scale * p.z + 10.0 * data.noise.perlin.turb(p, 7);
      return CudaColor(0.5, 0.5, 0.5) * (1.0 + sin(t));
    }

    case CudaTextureType::TEXTURE_CHECKER: {
      double inv_scale = 1.0 / data.checker.scale;
      int x = static_cast<int>(floor(inv_scale * p.x));
      int y = static_cast<int>(floor(inv_scale * p.y));
      int z = static_cast<int>(floor(inv_scale * p.z));
      bool is_even = (x + y + z) % 2 == 0;

      const void *tex_ptr =
          is_even ? data.checker.even_texture : data.checker.odd_texture;

      return cuda_texture_value(*reinterpret_cast<const CudaTexture *>(tex_ptr),
                                u, v, p);
    }
    }
    return CudaColor(0, 0, 0); // Should never reach here.
  }
};

// Helper to evaluate a texture value for a given CudaTexture struct.
__device__ inline CudaColor cuda_texture_value(const CudaTexture &tex, double u,
                                               double v, const CudaPoint3 &p) {
  return tex.value(u, v, p);
}

// Helper constructor functions.

__device__ inline CudaTexture cuda_make_solid_texture(CudaColor albedo) {
  CudaTexture texture;
  texture.type = CudaTextureType::TEXTURE_SOLID;
  texture.data.solid = CudaSolidColorTexture(albedo);
  return texture;
}

__device__ inline CudaTexture
cuda_make_checker_texture(double scale, const void *even_texture,
                          const void *odd_texture, CudaTextureType even_type,
                          CudaTextureType odd_type) {
  CudaTexture texture;
  texture.type = CudaTextureType::TEXTURE_CHECKER;
  texture.data.checker =
      CudaCheckerTexture(scale, even_texture, odd_texture, even_type, odd_type);
  return texture;
}

__device__ inline CudaTexture cuda_make_noise_texture(double scale,
                                                      CudaPerlinNoise perlin) {
  CudaTexture texture;
  texture.type = CudaTextureType::TEXTURE_NOISE;
  texture.data.noise = CudaNoiseTexture(scale, perlin);
  return texture;
}

#endif // USE_CUDA
