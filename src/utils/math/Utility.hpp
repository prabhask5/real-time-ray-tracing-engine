#pragma once

#include <cmath>
#include <limits>
#include <random>

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double PI = 3.1415926535897932385;

inline double degrees_to_radians(double degrees) {
  return degrees * PI / 180.0;
}

// Returns a random real in [0,1).
// Thread-local random number generator for deterministic concurrency.
inline std::mt19937 &random_engine() {
  static thread_local std::mt19937 generator{std::random_device{}()};
  return generator;
}

// Returns a random real in [0,1).
inline double random_double() {
  static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(random_engine());
}

// Returns a random real in [min,max).
inline double random_double(double min, double max) {
  std::uniform_real_distribution<double> dist(min, max);
  return dist(random_engine());
}

// Returns a random integer in [min,max].
inline int random_int(int min, int max) {
  std::uniform_int_distribution<int> dist(min, max);
  return dist(random_engine());
}
