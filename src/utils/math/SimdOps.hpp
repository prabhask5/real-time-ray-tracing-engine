#pragma once

#include "SimdTypes.hpp"
#include <cmath>

// Cross-platform SIMD operations abstraction.

// Provides unified interface for ARM NEON and x86 SSE/AVX operations.

namespace SimdOps {

inline simd_float4 load_float4(const float *ptr) {
#if SIMD_ARM_NEON
  return vld1q_f32(ptr);
#elif SIMD_X86_SSE
  return _mm_load_ps(ptr);
#else
  simd_float4 result;
  for (int i = 0; i < 4; ++i)
    result[i] = ptr[i];
  return result;
#endif
}

inline void store_float4(float *ptr, const simd_float4 &vec) {
#if SIMD_ARM_NEON
  vst1q_f32(ptr, vec);
#elif SIMD_X86_SSE
  _mm_store_ps(ptr, vec);
#else
  for (int i = 0; i < 4; ++i)
    ptr[i] = vec[i];
#endif
}

inline simd_float4 set_float4(float x, float y, float z, float w) {
#if SIMD_ARM_NEON
  const float data[4] = {x, y, z, w};
  return vld1q_f32(data);
#elif SIMD_X86_SSE
  return _mm_set_ps(w, z, y, x); // Note: x86 sets in reverse order.
#else
  simd_float4 result;
  result[0] = x;
  result[1] = y;
  result[2] = z;
  result[3] = w;
  return result;
#endif
}

inline simd_float4 add_float4(const simd_float4 &a, const simd_float4 &b) {
#if SIMD_ARM_NEON
  return vaddq_f32(a, b);
#elif SIMD_X86_SSE
  return _mm_add_ps(a, b);
#else
  simd_float4 result;
  for (int i = 0; i < 4; ++i)
    result[i] = a[i] + b[i];
  return result;
#endif
}

inline simd_float4 sub_float4(const simd_float4 &a, const simd_float4 &b) {
#if SIMD_ARM_NEON
  return vsubq_f32(a, b);
#elif SIMD_X86_SSE
  return _mm_sub_ps(a, b);
#else
  simd_float4 result;
  for (int i = 0; i < 4; ++i)
    result[i] = a[i] - b[i];
  return result;
#endif
}

inline simd_float4 mul_float4(const simd_float4 &a, const simd_float4 &b) {
#if SIMD_ARM_NEON
  return vmulq_f32(a, b);
#elif SIMD_X86_SSE
  return _mm_mul_ps(a, b);
#else
  simd_float4 result;
  for (int i = 0; i < 4; ++i)
    result[i] = a[i] * b[i];
  return result;
#endif
}

inline simd_float4 mul_scalar_float4(const simd_float4 &a, float scalar) {
#if SIMD_ARM_NEON
  return vmulq_n_f32(a, scalar);
#elif SIMD_X86_SSE
  return _mm_mul_ps(a, _mm_set1_ps(scalar));
#else
  simd_float4 result;
  for (int i = 0; i < 4; ++i)
    result[i] = a[i] * scalar;
  return result;
#endif
}

inline float dot_product_float4(const simd_float4 &a, const simd_float4 &b) {
#if SIMD_ARM_NEON
  simd_float4 mul = vmulq_f32(a, b);
  float32x2_t sum = vadd_f32(vget_low_f32(mul), vget_high_f32(mul));
  sum = vpadd_f32(sum, sum);
  return vget_lane_f32(sum, 0);
#elif SIMD_X86_SSE
#ifdef __SSE4_1__
  return _mm_cvtss_f32(_mm_dp_ps(a, b, 0x71));
#else
  __m128 mul = _mm_mul_ps(a, b);
  __m128 sum =
      _mm_add_ps(mul, _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 3, 0, 1)));
  sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2)));
  return _mm_cvtss_f32(sum);
#endif
#else
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
#endif
}

inline float length_squared_float4(const simd_float4 &a) {
  return dot_product_float4(a, a);
}

inline float length_float4(const simd_float4 &a) {
  return std::sqrt(length_squared_float4(a));
}

inline simd_float4 normalize_float4(const simd_float4 &a) {
  float len = length_float4(a);
  if (len > 1e-8f) {
    return mul_scalar_float4(a, 1.0f / len);
  }
  return set_float4(1.0f, 0.0f, 0.0f, 0.0f);
}

inline simd_double2 load_double2(const double *ptr) {
#if SIMD_ARM_NEON
  return vld1q_f64(ptr);
#elif SIMD_X86_SSE
  return _mm_load_pd(ptr);
#else
  simd_double2 result;
  result[0] = ptr[0];
  result[1] = ptr[1];
  return result;
#endif
}

inline void store_double2(double *ptr, const simd_double2 &vec) {
#if SIMD_ARM_NEON
  vst1q_f64(ptr, vec);
#elif SIMD_X86_SSE
  _mm_store_pd(ptr, vec);
#else
  ptr[0] = vec[0];
  ptr[1] = vec[1];
#endif
}

inline simd_double2 set_double2(double x, double y) {
#if SIMD_ARM_NEON
  const double data[2] = {x, y};
  return vld1q_f64(data);
#elif SIMD_X86_SSE
  return _mm_set_pd(y, x); // Note: x86 sets in reverse order.
#else
  simd_double2 result;
  result[0] = x;
  result[1] = y;
  return result;
#endif
}

inline simd_double2 add_double2(const simd_double2 &a, const simd_double2 &b) {
#if SIMD_ARM_NEON
  return vaddq_f64(a, b);
#elif SIMD_X86_SSE
  return _mm_add_pd(a, b);
#else
  simd_double2 result;
  result[0] = a[0] + b[0];
  result[1] = a[1] + b[1];
  return result;
#endif
}

inline simd_double2 sub_double2(const simd_double2 &a, const simd_double2 &b) {
#if SIMD_ARM_NEON
  return vsubq_f64(a, b);
#elif SIMD_X86_SSE
  return _mm_sub_pd(a, b);
#else
  simd_double2 result;
  result[0] = a[0] - b[0];
  result[1] = a[1] - b[1];
  return result;
#endif
}

inline simd_double2 mul_double2(const simd_double2 &a, const simd_double2 &b) {
#if SIMD_ARM_NEON
  return vmulq_f64(a, b);
#elif SIMD_X86_SSE
  return _mm_mul_pd(a, b);
#else
  simd_double2 result;
  result[0] = a[0] * b[0];
  result[1] = a[1] * b[1];
  return result;
#endif
}

inline simd_double2 mul_scalar_double2(const simd_double2 &a, double scalar) {
#if SIMD_ARM_NEON
  return vmulq_n_f64(a, scalar);
#elif SIMD_X86_SSE
  return _mm_mul_pd(a, _mm_set1_pd(scalar));
#else
  simd_double2 result;
  result[0] = a[0] * scalar;
  result[1] = a[1] * scalar;
  return result;
#endif
}

// Cross product for 3D vectors (using float4 with w=0).
inline simd_float4 cross_product_float4(const simd_float4 &a,
                                        const simd_float4 &b) {
#if SIMD_ARM_NEON
  // Cross product: (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x,
  // 0).
  float32x4_t a_yzxw =
      vextq_f32(vextq_f32(a, a, 3), a, 2); // a.y, a.z, a.x, a.w
  float32x4_t b_yzxw =
      vextq_f32(vextq_f32(b, b, 3), b, 2); // b.y, b.z, b.x, b.w
  float32x4_t a_zxyw =
      vextq_f32(vextq_f32(a, a, 2), a, 2); // a.z, a.x, a.y, a.w
  float32x4_t b_zxyw =
      vextq_f32(vextq_f32(b, b, 2), b, 2); // b.z, b.x, b.y, b.w

  simd_float4 result =
      vsubq_f32(vmulq_f32(a_yzxw, b_zxyw), vmulq_f32(a_zxyw, b_yzxw));
  // Set w component to 0.

  return vsetq_lane_f32(0.0f, result, 3);
#elif SIMD_X86_SSE
  // Cross product using SSE shuffles.

  __m128 a_yzx =
      _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)); // a.y, a.z, a.x, a.w
  __m128 b_yzx =
      _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1)); // b.y, b.z, b.x, b.w
  __m128 a_zxy =
      _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2)); // a.z, a.x, a.y, a.w
  __m128 b_zxy =
      _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2)); // b.z, b.x, b.y, b.w

  __m128 result =
      _mm_sub_ps(_mm_mul_ps(a_yzx, b_zxy), _mm_mul_ps(a_zxy, b_yzx));
  // Set w component to 0.

  return _mm_and_ps(result, _mm_castsi128_ps(_mm_set_epi32(
                                0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)));
#else
  simd_float4 result;
  result[0] = a[1] * b[2] - a[2] * b[1]; // x
  result[1] = a[2] * b[0] - a[0] * b[2]; // y
  result[2] = a[0] * b[1] - a[1] * b[0]; // z
  result[3] = 0.0f;                      // w
  return result;
#endif
}

} // namespace SimdOps