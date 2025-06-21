#pragma once

#include "SimdOps.hpp"
#include "SimdTypes.hpp"
#include "Vec3Utility.hpp"
#include <algorithm>

static const int PERLIN_POINT_COUNT = 256;

// This class implements Perlin noise. Perlin noise is a type of gradient noise.
// Instead of assigning random values to each point, it assigns random gradient
// Vectors to grid points and uses interpolation between them. This creates
// smooth, continuous noise (no harsh jumps like white noise).
// Memory layout optimized for noise generation performance.
class alignas(16) PerlinNoise {
public:
  PerlinNoise() {
    for (int i = 0; i < PERLIN_POINT_COUNT; i++)
      m_rand_vec[i] = unit_vector(Vec3::random(-1, 1));

    perlin_generate_perm(m_perm_x);
    perlin_generate_perm(m_perm_y);
    perlin_generate_perm(m_perm_z);
  }

  PerlinNoise(Vec3 rand_vec[PERLIN_POINT_COUNT], int perm_x[PERLIN_POINT_COUNT],
              int perm_y[PERLIN_POINT_COUNT], int perm_z[PERLIN_POINT_COUNT]) {
    std::copy(rand_vec, rand_vec + PERLIN_POINT_COUNT, m_rand_vec);
    std::copy(perm_x, perm_x + PERLIN_POINT_COUNT, m_perm_x);
    std::copy(perm_y, perm_y + PERLIN_POINT_COUNT, m_perm_y);
    std::copy(perm_z, perm_z + PERLIN_POINT_COUNT, m_perm_z);
  }

  // Generates a noise value at point p.
  double noise(const Point3 &p) const {
    double x_frac = p.x() - std::floor(p.x());
    double y_frac = p.y() - std::floor(p.y());
    double z_frac = p.z() - std::floor(p.z());

    int x_int = int(std::floor(p.x()));
    int y_int = int(std::floor(p.y()));
    int z_int = int(std::floor(p.z()));

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
  double turb(const Point3 &p, int depth) const {
    double accum = 0.0;
    Point3 temp_p = p;
    double weight = 1.0;

    for (int i = 0; i < depth; i++) {
      accum += weight * noise(temp_p);
      weight *= 0.5;
      temp_p *= 2;
    }

    return std::fabs(accum);
  }

  // Getter const methods.

  const Vec3 *rand_vec() const { return m_rand_vec; }

  const int *perm_x() const { return m_perm_x; }

  const int *perm_y() const { return m_perm_y; }

  const int *perm_z() const { return m_perm_z; }

private:
  // Hot data: most frequently accessed arrays for noise generation.

  // Random gradient vectors (accessed in noise()).
  Vec3 m_rand_vec[PERLIN_POINT_COUNT];

  // Warm data: permutation arrays grouped together for cache locality.

  // X-axis permutation array.
  int m_perm_x[PERLIN_POINT_COUNT];

  // Y-axis permutation array.
  int m_perm_y[PERLIN_POINT_COUNT];

  // Z-axis permutation array.
  int m_perm_z[PERLIN_POINT_COUNT];

  // Generates a permutation array used to shuffle access to gradients, ensuring
  // Pseudo-random yet deterministic behavior.
  void perlin_generate_perm(int *p) {
    for (int i = 0; i < PERLIN_POINT_COUNT; i++)
      p[i] = i;

    permute(p, PERLIN_POINT_COUNT);
  }

  // Shuffle around the values in a permutation array p, of length n.
  void permute(int *p, int n) {
    for (int i = n - 1; i > 0; i--) {
      int target = random_int(0, i);
      int tmp = p[i];
      p[i] = p[target];
      p[target] = tmp;
    }
  }

  // Interpolates dot products between the 8 corners of the unit cube:
  // - Uses Hermite cubic smoothing: u * u * (3 - 2 * u) to get a smooth noise
  // gradient.
  // - For each corner:
  // - - Compute weight_v = vector from corner to p.
  // - - Dot with gradient vector c[i][j][k].
  // - - Blend all 8 values together.
  double perlin_interp(const Vec3 c[2][2][2], double u, double v,
                       double w) const {
#if SIMD_AVAILABLE && SIMD_DOUBLE_PRECISION
    // SIMD-optimized Perlin interpolation for improved noise generation
    // performance.
    if constexpr (SIMD_DOUBLE_PRECISION) {
      double uu = u * u * (3 - 2 * u);
      double vv = v * v * (3 - 2 * v);
      double ww = w * w * (3 - 2 * w);

      // Process 8 corner calculations in parallel using SIMD.
      simd_double4 accum_vec = SimdOps::zero_double4();

      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          // Load weight vector components for both k=0 and k=1.
          simd_double4 weight_data =
              SimdOps::set_double4(u - i, v - j, w - 0, w - 1);

          // Process k=0 and k=1 together.
          Vec3 weight_v0(u - i, v - j, w - 0);
          Vec3 weight_v1(u - i, v - j, w - 1);

          double dot0 =
              c[i][j][0].dot(weight_v0); // SIMD-optimized dot product.
          double dot1 =
              c[i][j][1].dot(weight_v1); // SIMD-optimized dot product.

          double factor_i = (i * uu + (1 - i) * (1 - uu));
          double factor_j = (j * vv + (1 - j) * (1 - vv));

          double contrib0 =
              factor_i * factor_j * (0 * ww + (1 - 0) * (1 - ww)) * dot0;
          double contrib1 =
              factor_i * factor_j * (1 * ww + (1 - 1) * (1 - ww)) * dot1;

          simd_double4 contrib_vec =
              SimdOps::set_double4(contrib0, contrib1, 0, 0);
          accum_vec = SimdOps::add_double4(accum_vec, contrib_vec);
        }
      }

      // Sum the accumulated values.
      return accum_vec[0] + accum_vec[1];
    } else {
#endif
      double uu = u * u * (3 - 2 * u);
      double vv = v * v * (3 - 2 * v);
      double ww = w * w * (3 - 2 * w);
      double accum = 0.0;

      for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
          for (int k = 0; k < 2; k++) {
            Vec3 weight_v(u - i, v - j, w - k);
            accum += (i * uu + (1 - i) * (1 - uu)) *
                     (j * vv + (1 - j) * (1 - vv)) *
                     (k * ww + (1 - k) * (1 - ww)) *
                     dot_product(c[i][j][k], weight_v);
          }

      return accum;
#if SIMD_AVAILABLE && SIMD_DOUBLE_PRECISION
    }
#endif
  }
};