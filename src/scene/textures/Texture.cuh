#pragma once

#ifdef USE_CUDA

#include "../../core/Vec3Types.cuh"
#include "../../utils/math/PerlinNoise.cuh"
#include "../../utils/math/Vec3.cuh"

enum class CudaTextureType { TEXTURE_SOLID, TEXTURE_CHECKER, TEXTURE_NOISE };

// Forward declarations.
struct CudaTexture;

struct CudaSolidColorTexture {
  CudaColor albedo;

  __device__ CudaSolidColorTexture() {} // Default constructor.
  __host__ __device__ CudaSolidColorTexture(CudaColor _albedo)
      : albedo(_albedo) {}

  __device__ inline CudaColor value(double u, double v,
                                    const CudaPoint3 &p) const {
    return albedo;
  }
};

struct CudaCheckerTexture {
  double scale;
  const CudaTexture *even_texture;
  const CudaTexture *odd_texture;

  __device__ CudaCheckerTexture() {} // Default constructor.
  __host__ __device__ CudaCheckerTexture(double _scale,
                                         const CudaTexture *_even_texture,
                                         const CudaTexture *_odd_texture)
      : scale(_scale), even_texture(_even_texture), odd_texture(_odd_texture) {}

  __device__ CudaColor value(double u, double v, const CudaPoint3 &p) const;
};

struct CudaNoiseTexture {
  double scale;
  CudaPerlinNoise perlin;

  __device__ CudaNoiseTexture() {} // Default constructor.
  __host__ __device__ CudaNoiseTexture(double _scale, CudaPerlinNoise _perlin)
      : scale(_scale), perlin(_perlin) {}

  __device__ inline CudaColor value(double u, double v,
                                    const CudaPoint3 &p) const {
    // Calculates a procedural color value using sine-based noise, simulating
    // complex textures like marble. Uses a base gray color, with a sin function
    // to cause the brightness of the color to oscillate in a wavy pattern,
    // creating stripes or veins.

    return CudaColor(0.5, 0.5, 0.5) *
           (1 + sin(scale * p.z + 10 * perlin.turb(p, 7)));
  }
};

struct CudaTexture {
  CudaTextureType type;
  union {
    CudaSolidColorTexture *solid;
    CudaCheckerTexture *checker;
    CudaNoiseTexture *noise;
  };

  __host__ __device__ CudaTexture() {} // Default constructor.

  __device__ CudaColor value(double u, double v, const CudaPoint3 &p) const;
};

// Helper constructor functions.
__device__ CudaTexture cuda_make_solid_texture(CudaColor albedo);
__device__ CudaTexture
cuda_make_checker_texture(double scale, const CudaTexture *even_texture,
                          const CudaTexture *odd_texture);
__device__ CudaTexture cuda_make_noise_texture(double scale,
                                               CudaPerlinNoise perlin);

#endif // USE_CUDA
