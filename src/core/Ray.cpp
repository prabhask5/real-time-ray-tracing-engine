#include "Ray.hpp"
#include <Vec3Utility.hpp>

Ray::Ray() {}

Ray::Ray(const Point3 &origin, const Vec3 &direction)
    : m_origin(origin), m_direction(direction) {}

Ray::Ray(const Point3 &origin, const Vec3 &direction, double time)
    : m_origin(origin), m_direction(), m_time(time) {}

const Point3 &Ray::origin() const { return m_origin; }

const Vec3 &Ray::direction() const { return m_direction; }

const double Ray::time() const { return m_time; }

Point3 Ray::at(double t) const { return m_origin + t * m_direction; }