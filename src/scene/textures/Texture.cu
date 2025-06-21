#ifdef USE_CUDA

#include "Texture.cuh"

__device__ CudaColor CudaCheckerTexture::value(double u, double v,
                                               const CudaPoint3 &p) const {
  double inv_scale = 1.0 / scale;

  // Scale the point down to the texture's scale, in all three dimensions.
  int x_index = int(floor(inv_scale * p.x));
  int y_index = int(floor(inv_scale * p.y));
  int z_index = int(floor(inv_scale * p.z));

  // Use this to determine which checker space we're on.
  bool is_even = (x_index + y_index + z_index) % 2 == 0;

  return is_even ? even_texture->value(u, v, p) : odd_texture->value(u, v, p);
}

__device__ CudaColor CudaTexture::value(double u, double v,
                                        const CudaPoint3 &p) const {
  switch (type) {
  case CudaTextureType::TEXTURE_SOLID:
    return solid->value(u, v, p);

  case CudaTextureType::TEXTURE_NOISE:
    return noise->value(u, v, p);

  case CudaTextureType::TEXTURE_CHECKER:
    return checker->value(u, v, p);
  }
  return CudaColor(0, 0, 0); // Should never reach here.
}

__device__ CudaTexture cuda_make_solid_texture(CudaColor albedo) {
  CudaTexture texture;
  texture.type = CudaTextureType::TEXTURE_SOLID;
  texture.solid = new CudaSolidColorTexture(albedo);
  return texture;
}

__device__ CudaTexture
cuda_make_checker_texture(double scale, const CudaTexture *even_texture,
                          const CudaTexture *odd_texture) {
  CudaTexture texture;
  texture.type = CudaTextureType::TEXTURE_CHECKER;
  texture.checker = new CudaCheckerTexture(scale, even_texture, odd_texture);
  return texture;
}

__device__ CudaTexture cuda_make_noise_texture(double scale,
                                               CudaPerlinNoise perlin) {
  CudaTexture texture;
  texture.type = CudaTextureType::TEXTURE_NOISE;
  texture.noise = new CudaNoiseTexture(scale, perlin);
  return texture;
}

#endif // USE_CUDA