#include "Ray.hpp"
#include "../utils/math/Vec3Utility.hpp"

Ray::Ray() {}

Ray::Ray(const Point3 &origin, const Vec3 &direction)
    : m_origin(origin), m_direction(direction) {}

Ray::Ray(const Point3 &origin, const Vec3 &direction, double time)
    : m_origin(origin), m_direction(direction), m_time(time) {}

const Point3 &Ray::origin() const { return m_origin; }

const Vec3 &Ray::direction() const { return m_direction; }

double Ray::time() const { return m_time; }

// at() method moved to header file for SIMD optimization.