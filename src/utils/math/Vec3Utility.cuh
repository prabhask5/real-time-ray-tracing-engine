#pragma once

#ifdef USE_CUDA

#include "../../core/Vec3Types.cuh"
#include "Utility.cuh"
#include "Vec3.cuh"
#include <cmath>
#include <curand_kernel.h>

// Operator overloads for 3D vector.

__device__ inline CudaVec3 operator+(const CudaVec3 &u, const CudaVec3 &v) {
  return cuda_vec3_add(u, v);
}

__device__ inline CudaVec3 operator-(const CudaVec3 &u, const CudaVec3 &v) {
  return cuda_vec3_subtract(u, v);
}

__device__ inline CudaVec3 operator*(const CudaVec3 &u, const CudaVec3 &v) {
  return cuda_make_vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

__device__ inline CudaVec3 operator*(double t, const CudaVec3 &v) {
  return cuda_vec3_multiply_scalar(v, t);
}

__device__ inline CudaVec3 operator*(const CudaVec3 &v, double t) {
  return cuda_vec3_multiply_scalar(v, t);
}

__device__ inline CudaVec3 operator/(const CudaVec3 &v, double t) {
  return cuda_vec3_divide_scalar(v, t);
}

// Computes the dot product of two vectors.
__device__ inline double cuda_vec3_dot_product(const CudaVec3 &u,
                                               const CudaVec3 &v) {
  return u.x * v.x + u.y * v.y + u.z * v.z;
}

// Computes the cross product of two vectors.
__device__ inline CudaVec3 cuda_vec3_cross_product(const CudaVec3 &u,
                                                   const CudaVec3 &v) {
  return cuda_make_vec3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z,
                        u.x * v.y - u.y * v.x);
}

// Returns the unit-length (normalized) vector.
__device__ inline CudaVec3 cuda_vec3_unit_vector(const CudaVec3 &v) {
  return cuda_vec3_divide_scalar(v, cuda_vec3_length(v));
}

// Optimized random point in unit disk using polar coordinates.
// More efficient than rejection sampling for GPU.
__device__ inline CudaPoint3 cuda_vec3_random_in_unit_disk(curandState *state) {
  double r = sqrt(cuda_random_double(state));
  double theta = cuda_random_double(state, 0.0, 2.0 * CUDA_PI);
  return cuda_make_vec3(r * cos(theta), r * sin(theta), 0.0);
}

// Optimized random unit vector generation using spherical coordinates.
// More efficient than rejection sampling for GPU.
__device__ inline CudaVec3 cuda_vec3_random_unit_vector(curandState *state) {
  double z = cuda_random_double(state, -1.0, 1.0);
  double a = cuda_random_double(state, 0.0, 2.0 * CUDA_PI);
  double r = sqrt(1.0 - z * z);
  return cuda_make_vec3(r * cos(a), r * sin(a), z);
}

// Generates a random unit vector that lies on the same hemisphere as a given
// normal vector.
__device__ inline CudaVec3
cuda_vec3_random_on_hemisphere(curandState *state, const CudaVec3 &normal) {
  CudaVec3 on_unit_sphere = cuda_vec3_random_unit_vector(state);
  if (cuda_vec3_dot_product(on_unit_sphere, normal) >
      0.0) // In the same hemisphere as the normal.
    return on_unit_sphere;
  else
    return cuda_vec3_negate(on_unit_sphere);
}

// Computes the reflection of incident vector off a surface with normal vector.
__device__ inline CudaVec3 cuda_vec3_reflect(const CudaVec3 &incident,
                                             const CudaVec3 &normal) {
  return cuda_vec3_subtract(
      incident, cuda_vec3_multiply_scalar(
                    normal, 2 * cuda_vec3_dot_product(incident, normal)));
}

// Computes the refraction (bending) of a light ray uv passing through a surface
// with normal n and relative refractive index etai_over_etat.
__device__ inline CudaVec3 cuda_vec3_refract(const CudaVec3 &ray,
                                             const CudaVec3 &normal,
                                             double etai_over_etat) {
  double cos_theta =
      fmin(cuda_vec3_dot_product(cuda_vec3_negate(ray), normal), 1.0);
  CudaVec3 r_out_perp = cuda_vec3_multiply_scalar(
      cuda_vec3_add(ray, cuda_vec3_multiply_scalar(normal, cos_theta)),
      etai_over_etat);
  CudaVec3 r_out_parallel = cuda_vec3_multiply_scalar(
      normal, -sqrt(fabs(1.0 - cuda_vec3_length_squared(r_out_perp))));
  return cuda_vec3_add(r_out_perp, r_out_parallel);
}

// Generates a random unit vector that is biased toward the +z direction,
// following a cosine-weighted distribution, commonly used in physically-based
// rendering (PBR) to simulate diffuse reflection.
__device__ inline CudaVec3
cuda_vec3_random_cosine_direction(curandState *state) {
  double r1 = cuda_random_double(state);
  double r2 = cuda_random_double(state);

  double phi = 2 * CUDA_PI * r1;
  double x = cos(phi) * sqrt(r2);
  double y = sin(phi) * sqrt(r2);
  double z = sqrt(1.0 - r2);

  return cuda_make_vec3(x, y, z);
}

#endif // USE_CUDA
