#pragma once

#include <Interval.hpp>

Interval operator+(const Interval &ival, double displacement) {
  return Interval(ival.min() + displacement, ival.max() + displacement);
}

Interval operator+(double displacement, const Interval &ival) {
  return ival + displacement;
}