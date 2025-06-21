#pragma once

#include "SimdOps.hpp"
#include "SimdTypes.hpp"
#include "Utility.hpp"

// SIMD-optimized Interval class with optimal memory layout and SIMD operations.
class alignas(16) Interval {
public:
  Interval() : m_min(+INF), m_max(-INF) {} // Simulates empty interval.

  Interval(double min, double max) : m_min(min), m_max(max) {}

  // Create the interval tightly enclosing the two input intervals with SIMD
  // acceleration.
  Interval(const Interval &a, const Interval &b) {
#if SIMD_AVAILABLE && SIMD_DOUBLE_PRECISION
    // Use SIMD for min/max operations when available.

    alignas(16) double a_data[2] = {a.m_min, a.m_max};
    alignas(16) double b_data[2] = {b.m_min, b.m_max};
    alignas(16) double result_data[2];

    simd_double2 a_vec = SimdOps::load_double2(a_data);
    simd_double2 b_vec = SimdOps::load_double2(b_data);

    // Min operation for first element, max operation for second element.
    result_data[0] = (a.m_min <= b.m_min) ? a.m_min : b.m_min;
    result_data[1] = (a.m_max >= b.m_max) ? a.m_max : b.m_max;

    m_min = result_data[0];
    m_max = result_data[1];
#else
    m_min = a.min() <= b.min() ? a.min() : b.min();
    m_max = a.max() >= b.max() ? a.max() : b.max();
#endif
  }

  // Getter const methods.

  double min() const { return m_min; }

  double max() const { return m_max; }

  double size() const { return m_max - m_min; }

  bool contains(double x) const { return m_min <= x && x <= m_max; }

  bool surrounds(double x) const { return m_min < x && x < m_max; }

  double clamp(double x) const {
    if (x < m_min)
      return m_min;
    if (x > m_max)
      return m_max;
    return x;
  }

  // Make a new interval that is expanded by the delta value with SIMD
  // acceleration.
  Interval expand(double delta) const {
    double padding = delta * 0.5;
#if SIMD_AVAILABLE && SIMD_DOUBLE_PRECISION
    // Use SIMD for simultaneous subtract and add operations.
    alignas(16) double current[2] = {m_min, m_max};
    alignas(16) double padding_vec[2] = {-padding, padding};
    alignas(16) double result[2];

    simd_double2 current_simd = SimdOps::load_double2(current);
    simd_double2 padding_simd = SimdOps::load_double2(padding_vec);
    simd_double2 result_simd = SimdOps::add_double2(current_simd, padding_simd);
    SimdOps::store_double2(result, result_simd);

    return Interval(result[0], result[1]);
#else
    return Interval(m_min - padding, m_max + padding);
#endif
  }

  void set_interval(double min, double max) {
    m_min = min;
    m_max = max;
  }

private:
  double m_min;
  double m_max;
};

static inline const Interval EMPTY_INTERVAL = Interval(+INF, -INF);
static inline const Interval UNIVERSE_INTERVAL = Interval(-INF, +INF);