#ifdef USE_CUDA

#include "Texture.cuh"

__device__ CudaColor
cuda_checker_texture_value(const CudaCheckerTexture &texture, double u,
                           double v, const CudaPoint3 &p) {
  double inv_scale = 1.0 / texture.scale;

  int x_index = int(floor(inv_scale * p.x));
  int y_index = int(floor(inv_scale * p.y));
  int z_index = int(floor(inv_scale * p.z));

  bool is_even = (x_index + y_index + z_index) % 2 == 0;

  return is_even ? cuda_texture_value(*texture.even_texture, u, v, p)
                 : cuda_texture_value(*texture.odd_texture, u, v, p);
}

__device__ CudaColor cuda_texture_value(const CudaTexture &texture, double u,
                                        double v, const CudaPoint3 &p) {
  switch (texture.type) {
  case CudaTextureType::TEXTURE_SOLID:
    return cuda_solid_color_texture_value(*texture.solid, u, v, p);
  case CudaTextureType::TEXTURE_NOISE:
    return cuda_noise_texture_value(*texture.noise, u, v, p);
  case CudaTextureType::TEXTURE_CHECKER:
    return cuda_checker_texture_value(*texture.checker, u, v, p);
  default:
    // ERROR: Texture.cu::cuda_texture_value - Unknown Texture type in switch
    // statement. This should never happen in well-formed code.
    return cuda_make_vec3(0.0, 0.0, 0.0); // Safe fallback for GPU device code.
  }
}

__device__ CudaTexture cuda_make_texture_solid(CudaColor albedo) {
  CudaTexture texture;
  texture.type = CudaTextureType::TEXTURE_SOLID;
  texture.solid = new CudaSolidColorTexture();
  *texture.solid = cuda_make_solid_color_texture(albedo);
  return texture;
}

__device__ CudaTexture
cuda_make_texture_checker(double scale, const CudaTexture *even_texture,
                          const CudaTexture *odd_texture) {
  CudaTexture texture;
  texture.type = CudaTextureType::TEXTURE_CHECKER;
  texture.checker = new CudaCheckerTexture();
  *texture.checker =
      cuda_make_checker_texture(scale, even_texture, odd_texture);
  return texture;
}

__device__ CudaTexture cuda_make_texture_noise(double scale,
                                               CudaPerlinNoise perlin) {
  CudaTexture texture;
  texture.type = CudaTextureType::TEXTURE_NOISE;
  texture.noise = new CudaNoiseTexture();
  *texture.noise = cuda_make_noise_texture(scale, perlin);
  return texture;
}

#endif // USE_CUDA