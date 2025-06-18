#pragma once

#ifdef USE_CUDA

#include "Vec3Utility.cuh"
#include <curand_kernel.h>

// Number of random gradient vectors and permutations.
static const int PERLIN_POINT_COUNT = 256;

// This struct implements Perlin noise for CUDA. It holds gradient vectors and
// permutation tables.
struct CudaPerlinNoise {
  CudaVec3 rand_vec[PERLIN_POINT_COUNT];
  int perm_x[PERLIN_POINT_COUNT];
  int perm_y[PERLIN_POINT_COUNT];
  int perm_z[PERLIN_POINT_COUNT];
};

// Generates a permutation array used to shuffle access to gradients, ensuring
// pseudo-random yet deterministic behavior.
__device__ __forceinline__ void cuda_perlin_generate_perm(int *p) {
  for (int i = 0; i < PERLIN_POINT_COUNT; i++)
    p[i] = i;
  for (int i = PERLIN_POINT_COUNT - 1; i > 0; i--) {
    int target = curand(&p[0]) % (i + 1); // Pseudo-shuffle with curand.
    int tmp = p[i];
    p[i] = p[target];
    p[target] = tmp;
  }
}

// Initializes the Perlin noise object by generating gradient vectors and
// permutations.
__device__ __forceinline__ void cuda_init_perlin(CudaPerlinNoise *pn,
                                                 curandState *state) {
  for (int i = 0; i < PERLIN_POINT_COUNT; i++) {
    pn->rand_vec[i] =
        cuda_unit_vector(CudaVec3(curand_uniform_double(state) * 2.0 - 1.0,
                                  curand_uniform_double(state) * 2.0 - 1.0,
                                  curand_uniform_double(state) * 2.0 - 1.0));
  }
  cuda_perlin_generate_perm(pn->perm_x);
  cuda_perlin_generate_perm(pn->perm_y);
  cuda_perlin_generate_perm(pn->perm_z);
}

// Interpolates dot products between the 8 corners of the unit cube:
// - Uses Hermite cubic smoothing: u * u * (3 - 2 * u)
// - For each corner:
//   - Compute weight_v = vector from corner to p
//   - Dot with gradient vector c[i][j][k]
//   - Blend all 8 values together
__device__ __forceinline__ double
cuda_perlin_interp(const CudaVec3 c[2][2][2], double u, double v, double w) {
  double uu = u * u * (3 - 2 * u);
  double vv = v * v * (3 - 2 * v);
  double ww = w * w * (3 - 2 * w);
  double accum = 0.0;

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++) {
        CudaVec3 weight_v(u - i, v - j, w - k);
        accum += (i * uu + (1 - i) * (1 - uu)) * (j * vv + (1 - j) * (1 - vv)) *
                 (k * ww + (1 - k) * (1 - ww)) *
                 cuda_dot_product(c[i][j][k], weight_v);
      }

  return accum;
}

// Generates a noise value at point p.
__device__ __forceinline__ double cuda_perlin_noise(const CudaPerlinNoise *pn,
                                                    const CudaPoint3 &p) {
  double x_frac = p.x() - floor(p.x());
  double y_frac = p.y() - floor(p.y());
  double z_frac = p.z() - floor(p.z());

  int x_int = static_cast<int>(floor(p.x()));
  int y_int = static_cast<int>(floor(p.y()));
  int z_int = static_cast<int>(floor(p.z()));

  CudaVec3 c[2][2][2];

  for (int di = 0; di < 2; di++)
    for (int dj = 0; dj < 2; dj++)
      for (int dk = 0; dk < 2; dk++) {
        int idx = pn->perm_x[(x_int + di) & 255] ^
                  pn->perm_y[(y_int + dj) & 255] ^
                  pn->perm_z[(z_int + dk) & 255];
        c[di][dj][dk] = pn->rand_vec[idx];
      }

  return cuda_perlin_interp(c, x_frac, y_frac, z_frac);
}

// Generates turbulant noise, used for visual complexity like smoke or clouds.
__device__ __forceinline__ double cuda_perlin_turb(const CudaPerlinNoise *pn,
                                                   CudaPoint3 p, int depth) {
  double accum = 0.0;
  double weight = 1.0;

  for (int i = 0; i < depth; i++) {
    accum += weight * cuda_perlin_noise(pn, p);
    weight *= 0.5;
    p *= 2.0;
  }

  return fabs(accum);
}

#endif // USE_CUDA
