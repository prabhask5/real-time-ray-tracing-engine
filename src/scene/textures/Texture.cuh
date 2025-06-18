#pragma once

#ifdef USE_CUDA

#include "../../core/Vec3Types.cuh"
#include "../../utils/math/PerlinNoise.cuh"
#include "../../utils/math/Vec3.cuh"

enum class CudaTextureType { Solid, Checker, Noise };

struct CudaSolidColorTexture {
  CudaColor albedo;
};

struct CudaCheckerTexture {
  double scale;
  const void *even_texture;
  const void *odd_texture;
  CudaTextureType even_type;
  CudaTextureType odd_type;
};

struct CudaNoiseTexture {
  double scale;
  CudaPerlinNoise perlin;
};

struct CudaTexture {
  CudaTextureType type;
  union {
    CudaSolidColorTexture solid;
    CudaCheckerTexture checker;
    CudaNoiseTexture noise;
  };

  __device__ inline CudaColor value(double u, double v, const CudaPoint3 &p) {
    switch (type) {
    case CudaTextureType::Solid:
      return solid.albedo;

    case CudaTextureType::Noise: {
      double t = noise.scale * p.z + 10.0 * noise.perlin.turb(p, 7);
      return CudaColor(0.5, 0.5, 0.5) * (1.0 + sin(t));
    }

    case CudaTextureType::Checker: {
      double inv_scale = 1.0 / checker.scale;
      int x = static_cast<int>(floor(inv_scale * p.x));
      int y = static_cast<int>(floor(inv_scale * p.y));
      int z = static_cast<int>(floor(inv_scale * p.z));
      bool is_even = (x + y + z) % 2 == 0;

      const void *tex_ptr =
          is_even ? checker.even_texture : checker.odd_texture;
      CudaTextureType tex_type = is_even ? checker.even_type : checker.odd_type;

      return cuda_texture_value(*reinterpret_cast<const CudaTexture *>(tex_ptr),
                                u, v, p);
    }
    }
    return CudaColor(0, 0, 0); // Should never reach here.
  }
};

#endif // USE_CUDA
