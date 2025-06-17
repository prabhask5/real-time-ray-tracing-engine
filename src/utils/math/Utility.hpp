#pragma once

#include <cmath>
#include <cstdlib>
#include <limits>

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double PI = 3.1415926535897932385;

inline double degrees_to_radians(double degrees) {
  return degrees * PI / 180.0;
}

// Returns a random real in [0,1).
inline double random_double() { return std::rand() / (RAND_MAX + 1.0); }

// Returns a random real in [min,max).
inline double random_double(double min, double max) {
  return min + (max - min) * random_double();
}

// Returns a random integer in [min,max].
inline int random_int(int min, int max) {
  return int(random_double(min, max + 1));
}
