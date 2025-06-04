#include "Sphere.hpp"
#include "../core/AABB.hpp"
#include "../core/HitRecord.hpp"
#include "../core/Ray.hpp"
#include <Interval.hpp>
#include <Vec3Utility.hpp>

Sphere::Sphere(const Point3 &center, double radius, MaterialPtr material)
    : m_center(center), m_radius(radius), m_material(material) {}

bool Sphere::hit(const Ray &ray, Interval t_values, HitRecord &record) const {
  // Let the ray be: r(t) = origin + t * direction
  // We want to find t such that: [r(t) - center]^2 = radius^2
  // This yields a quadratic: a*t^2 - 2h*t + c = 0
  // Where: oc = center - origin
  // a = direction.length_squared()
  // h = dot(direction, oc)
  // c = [oc]^2 - r^2
  Vec3 oc = m_center - ray.origin();
  double a = ray.direction().length_squared();
  double h = dot_product(ray.direction(), oc);
  double c = oc.length_squared() - m_radius * m_radius;

  // Find the discriminant to easily see if there's any intersection points.
  double discriminant = h * h - a * c;
  if (discriminant < 0)
    return false;

  double sqrtd = std::sqrt(discriminant);

  // Find the nearest root that lies in the acceptable range.
  double root = (h - sqrtd) / a; // Via the quadratic equation.
  if (!t_values.surrounds(root)) {
    root = (h + sqrtd) / a; // Other root.
    if (!t_values.surrounds(root))
      return false; // Seems like none exist then.
  }

  // Populate the hit record.
  record.t = root;
  record.point = ray.at(record.t);
  Vec3 outward_normal = (record.point - m_center) / m_radius;
  record.set_face_normal(ray, outward_normal);
  record.material = m_material;

  return true;
}

bool Sphere::bounding_box(AABB &output_box) const {
  output_box = AABB(m_center - Vec3(m_radius, m_radius, m_radius),
                    m_center + Vec3(m_radius, m_radius, m_radius));
  return true;
}