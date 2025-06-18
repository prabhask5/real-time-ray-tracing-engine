#pragma once

#include "Vec3Utility.hpp"

static const int PERLIN_POINT_COUNT = 256;

// This class implements Perlin noise. Perlin noise is a type of gradient noise.
// Instead of assigning random values to each point, it assigns random gradient
// Vectors to grid points and uses interpolation between them. This creates
// smooth, continuous noise (no harsh jumps like white noise).
class PerlinNoise {
public:
  PerlinNoise() {
    for (int i = 0; i < PERLIN_POINT_COUNT; i++)
      m_rand_vec[i] = unit_vector(Vec3::random(-1, 1));

    perlin_generate_perm(m_perm_x);
    perlin_generate_perm(m_perm_y);
    perlin_generate_perm(m_perm_z);
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

private:
  Vec3 m_rand_vec[PERLIN_POINT_COUNT];
  int m_perm_x[PERLIN_POINT_COUNT];
  int m_perm_y[PERLIN_POINT_COUNT];
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
  // Gradient.
  // - For each corner:
  // - - Compute weight_v = vector from corner to p
  // - - Dot with gradient vector c[i][j][k]
  // - - Blend all 8 values together
  double perlin_interp(const Vec3 c[2][2][2], double u, double v,
                       double w) const {
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
              (k * ww + (1 - k) * (1 - ww)) * dot_product(c[i][j][k], weight_v);
        }

    return accum;
  }
};