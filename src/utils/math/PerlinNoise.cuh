#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "Vec3Utility.cuh"

static const int PERLIN_POINT_COUNT = 256;

// This class implements Perlin noise. Perlin noise is a type of gradient noise.
// Instead of assigning random values to each point, it assigns random gradient
// vectors to grid points and uses interpolation between them. This creates
// smooth, continuous noise (no harsh jumps like white noise).
class PerlinNoise {
public:
  __device__ PerlinNoise() {}

  __device__ PerlinNoise(Vec3* rand_vec, int* perm_x, int* perm_y, int* perm_z) :
    m_rand_vec(rand_vec), m_perm_x(perm_x), m_perm_y(perm_y), m_perm_z(perm_z) {}

  // Generates a noise value at point p.
  __device__ double noise(const Point3& p) const {
    double x_frac = p.x() - floor(p.x());
    double y_frac = p.y() - floor(p.y());
    double z_frac = p.z() - floor(p.z());

    int x_int = int(floor(p.x()));
    int y_int = int(floor(p.y()));
    int z_int = int(floor(p.z()));

    // Stores gradient vectors at the 8 corners of the cube surrounding point p.
    Vec3 c[2][2][2];

    // Hash the permutation values to get the gradient vectors.
    for (int di = 0; di < 2; di++)
      for (int dj = 0; dj < 2; dj++)
        for (int dk = 0; dk < 2; dk++)
          c[di][dj][dk] = m_rand_vec[m_perm_x[(x_int + di) & 255] ^
                                     m_perm_y[(y_int + dj) & 255] ^
                                     m_perm_z[(z_int + dk) & 255]];

    return perlin_interp(c, x_frac, y_frac, z_frac);
  }

  // Generates turbulant noise, used for visual complexity like smoke or clouds.
  __device__ double turb(const Point3& p, int depth) const {
    double accum = 0.0;
    Point3 temp_p = p;
    double weight = 1.0;

    for (int i = 0; i < depth; i++) {
      accum += weight * noise(temp_p);
      weight *= 0.5;
      temp_p *= 2;
    }

    return fabs(accum);
  }

private:
  Vec3* m_rand_vec;
  int* m_perm_x;
  int* m_perm_y;
  int* m_perm_z;

  // Interpolates dot products between the 8 corners of the unit cube:
  // - Uses Hermite cubic smoothing: u * u * (3 - 2 * u) to get a smooth noise
  // gradient.
  // - For each corner:
  // - - Compute weight_v = vector from corner to p
  // - - Dot with gradient vector c[i][j][k]
  // - - Blend all 8 values together
  __device__ double perlin_interp(const Vec3 c[2][2][2], double u, double v, double w) const {
    double uu = u * u * (3 - 2 * u);
    double vv = v * v * (3 - 2 * v);
    double ww = w * w * (3 - 2 * w);
    double accum = 0.0;

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) {
          Vec3 weight_v(u - i, v - j, w - k);
          accum +=
              (i * uu + (1 - i) * (1 - uu)) * (j * vv + (1 - j) * (1 - vv)) *
              (k * ww + (1 - k) * (1 - ww)) * dot(c[i][j][k], weight_v);
        }

    return accum;
  }
};

// Host function to allocate and initialize Perlin noise data on device.
inline void cuda_allocate_perlin_noise(Vec3** d_rand_vec, int** d_perm_x, int** d_perm_y, int** d_perm_z) {
    // Allocate device memory.
    cudaMalloc(d_rand_vec, PERLIN_POINT_COUNT * sizeof(Vec3));
    cudaMalloc(d_perm_x, PERLIN_POINT_COUNT * sizeof(int));
    cudaMalloc(d_perm_y, PERLIN_POINT_COUNT * sizeof(int));
    cudaMalloc(d_perm_z, PERLIN_POINT_COUNT * sizeof(int));
    
    // Create host-side data.
    Vec3 host_rand_vec[PERLIN_POINT_COUNT];
    int host_perm_x[PERLIN_POINT_COUNT];
    int host_perm_y[PERLIN_POINT_COUNT];
    int host_perm_z[PERLIN_POINT_COUNT];
    
    // Generate random vectors (mirroring CPU initialization).
    for (int i = 0; i < PERLIN_POINT_COUNT; i++) {
        host_rand_vec[i] = unit_vector(Vec3::random(-1, 1));
    }
    
    // Initialize permutation arrays.
    for (int i = 0; i < PERLIN_POINT_COUNT; i++) {
        host_perm_x[i] = i;
        host_perm_y[i] = i;
        host_perm_z[i] = i;
    }
    
    // Shuffle permutation arrays (Fisher-Yates shuffle).
    for (int i = PERLIN_POINT_COUNT - 1; i > 0; i--) {
        int target_x = random_int(0, i);
        int target_y = random_int(0, i);
        int target_z = random_int(0, i);
        
        // Swap X.
        int temp = host_perm_x[i];
        host_perm_x[i] = host_perm_x[target_x];
        host_perm_x[target_x] = temp;
        
        // Swap Y.
        temp = host_perm_y[i];
        host_perm_y[i] = host_perm_y[target_y];
        host_perm_y[target_y] = temp;
        
        // Swap Z.
        temp = host_perm_z[i];
        host_perm_z[i] = host_perm_z[target_z];
        host_perm_z[target_z] = temp;
    }
    
    // Copy to device.
    cudaMemcpy(*d_rand_vec, host_rand_vec, PERLIN_POINT_COUNT * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_perm_x, host_perm_x, PERLIN_POINT_COUNT * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_perm_y, host_perm_y, PERLIN_POINT_COUNT * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_perm_z, host_perm_z, PERLIN_POINT_COUNT * sizeof(int), cudaMemcpyHostToDevice);
}

// Host function to free Perlin noise device memory.
inline void cuda_free_perlin_noise(Vec3* d_rand_vec, int* d_perm_x, int* d_perm_y, int* d_perm_z) {
    if (d_rand_vec) cudaFree(d_rand_vec);
    if (d_perm_x) cudaFree(d_perm_x);
    if (d_perm_y) cudaFree(d_perm_y);
    if (d_perm_z) cudaFree(d_perm_z);
}

#endif // USE_CUDA
