#pragma once

// SIMD abstraction layer for cross-platform vector operations.

// Automatically detects and uses ARM NEON or x86 SSE/AVX intrinsics.

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define SIMD_ARM_NEON 1
#define SIMD_X86_SSE 0
#elif defined(__SSE__) || defined(__SSE2__) || defined(__AVX__) ||             \
    defined(_M_X64) || defined(_M_IX86_FP)
#include <immintrin.h>
#define SIMD_ARM_NEON 0
#define SIMD_X86_SSE 1
#else
#define SIMD_ARM_NEON 0
#define SIMD_X86_SSE 0
#endif

// SIMD vector type abstraction.

#if SIMD_ARM_NEON
using simd_float4 = float32x4_t;
using simd_double2 = float64x2_t;
#elif SIMD_X86_SSE
using simd_float4 = __m128;
using simd_double2 = __m128d;
#else
// Fallback: plain array for platforms without SIMD.

struct alignas(16) simd_float4 {
  float data[4];
  float &operator[](int i) { return data[i]; }
  const float &operator[](int i) const { return data[i]; }
};
struct alignas(16) simd_double2 {
  double data[2];
  double &operator[](int i) { return data[i]; }
  const double &operator[](int i) const { return data[i]; }
};
#endif

// Feature detection constants.

constexpr bool SIMD_AVAILABLE = SIMD_ARM_NEON || SIMD_X86_SSE;
constexpr bool SIMD_DOUBLE_PRECISION =
    true; // Both ARM and x86 support double precision SIMD.