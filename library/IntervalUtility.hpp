#pragma once

#include "Interval.hpp"

inline Interval operator+(const Interval &ival, double displacement) {
  return Interval(ival.min() + displacement, ival.max() + displacement);
}

inline Interval operator+(double displacement, const Interval &ival) {
  return ival + displacement;
}