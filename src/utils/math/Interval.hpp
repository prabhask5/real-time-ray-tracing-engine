#pragma once

#include "Utility.hpp"

// SIMD-optimized Interval class with optimal memory layout.
class alignas(16) Interval {
public:
  Interval() : m_min(+INF), m_max(-INF) {} // Simulates empty interval.

  Interval(double min, double max) : m_min(min), m_max(max) {}

  // Create the interval tightly enclosing the two input intervals.
  Interval(const Interval &a, const Interval &b) {
    m_min = a.min() <= b.min() ? a.min() : b.min();
    m_max = a.max() >= b.max() ? a.max() : b.max();
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

  // Make a new interval that is expanded by the delta value.
  Interval expand(double delta) const {
    double padding = delta / 2;
    return Interval(m_min - padding, m_max + padding);
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