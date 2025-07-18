#pragma once

#include "../core/Vec3Types.hpp"
#include "../utils/math/Interval.hpp"
#include "../utils/math/SimdOps.hpp"
#include "../utils/math/SimdTypes.hpp"
#include <iomanip>
#include <sstream>

class Ray; // From Ray.hpp.

// Defines an Axis-Aligned Bounding Box: the simplest type of bounding box: a
// box aligned with the coordinate axes, defined by its minimum and maximum
// points in 3D space. Checking ray-box intersections is much faster than
// ray-triangle or ray-sphere intersections. So you wrap objects in AABBs and
// check the ray against those first.
// Cache-optimized layout for fast ray-box intersection tests.
class alignas(16) AABB {
public:
  AABB();

  AABB(const Interval &x, const Interval &y, const Interval &z);

  // Make the box using the two points at the diagonal boundaries.
  AABB(const Point3 &p1, const Point3 &p2);

  AABB(const AABB &b1, const AABB &b2);

  // Getter const methods.

  const Interval &x() const;

  const Interval &y() const;

  const Interval &z() const;

  const Interval &get_axis_interval(int index) const;

  // Returns the index of the longest axis of the bounding box.
  int get_longest_axis() const;

  // Returns the center point of the bounding box.
  Point3 center() const;

  // Returns the surface area of the bounding box.
  double surface_area() const;

  // Action methods.

  // This function checks if the ray hits the hittable object with the t values
  // in the interval range ray_t.
  bool hit(const Ray &r, Interval t_values) const;

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"AABB\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"x\":" << m_x.json() << ",";
    oss << "\"y\":" << m_y.json() << ",";
    oss << "\"z\":" << m_z.json();
    oss << "}";
    return oss.str();
  }

private:
  // Adjust the AABB so that no side is narrower than some delta, padding if
  // necessary.
  void pad_to_minimums();

private:
  // Box range for the x-axis.
  Interval m_x;

  // Box range for the y-axis.
  Interval m_y;

  // Box range for the z-axis.
  Interval m_z;
};

static inline const AABB EMPTY_AABB =
    AABB(EMPTY_INTERVAL, EMPTY_INTERVAL, EMPTY_INTERVAL);
static inline const AABB UNIVERSE_AABB =
    AABB(UNIVERSE_INTERVAL, UNIVERSE_INTERVAL, UNIVERSE_INTERVAL);