#include "Sphere.hpp"
#include "../../core/HitRecord.hpp"
#include "../../core/Ray.hpp"
#include "../../utils/math/Interval.hpp"
#include "../../utils/math/ONB.hpp"
#include "../../utils/math/Vec3Utility.hpp"

Sphere::Sphere(const Point3 &center, double radius, MaterialPtr material)
    : m_center(center, Vec3(0, 0, 0)), m_radius(std::fmax(0, radius)),
      m_material(material) {
  Vec3 radius_vector = Vec3(radius, radius, radius);
  m_bbox = AABB(center - radius_vector, center + radius_vector);
}

Sphere::Sphere(const Point3 &before_center, const Point3 &after_center,
               double radius, MaterialPtr material)
    : m_center(before_center, after_center - before_center),
      m_radius(std::fmax(0, radius)), m_material(material) {
  Vec3 radius_vector = Vec3(radius, radius, radius);
  AABB box1(m_center.at(0) - radius_vector, m_center.at(0) + radius_vector);
  AABB box2(m_center.at(1) - radius_vector, m_center.at(1) + radius_vector);
  m_bbox = AABB(box1, box2);
}

AABB Sphere::get_bounding_box() const { return m_bbox; }

bool Sphere::hit(const Ray &ray, Interval t_values, HitRecord &record) const {
  // Let the ray be: r(t) = origin + t * direction
  // We want to find t such that: [r(t) - center]^2 = radius^2
  // This yields a quadratic: a*t^2 - 2h*t + c = 0
  // Where: oc = center - origin
  // a = direction.length_squared()
  // h = dot(direction, oc)
  // c = [oc]^2 - r^2
  Point3 current_center = m_center.at(ray.time());
  Vec3 oc = current_center - ray.origin();
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
  Vec3 outward_normal = (record.point - current_center) / m_radius;
  record.set_face_normal(ray, outward_normal);
  record.material = m_material;

  // outward_normal: a given point on the sphere of radius one, centered at the
  // origin. record.u: returned value [0,1] of angle around the Y axis from
  // X=-1. record.v: returned value [0,1] of angle from Y=-1 to Y=+1.
  //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
  //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
  //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

  double theta = std::acos(-outward_normal.y());
  double phi = std::atan2(-outward_normal.z(), outward_normal.x()) + PI;

  record.u = phi / (2 * PI);
  record.v = theta / PI;

  return true;
}

double Sphere::pdf_value(const Point3 &origin, const Vec3 &direction) const {
  // This method only works for stationary spheres.

  HitRecord record;
  if (!this->hit(Ray(origin, direction), Interval(0.001, INF), record))
    return 0;

  double dist_squared = (m_center.at(0) - origin).length_squared();
  double cos_theta_max = std::sqrt(1 - m_radius * m_radius / dist_squared);
  double solid_angle = 2 * PI * (1 - cos_theta_max);

  // PDF Formula for solid angle sampling.
  return 1 / solid_angle;
}

Vec3 Sphere::random(const Point3 &origin) const {
  Vec3 direction = m_center.at(0) - origin;
  double distance_squared = direction.length_squared();
  ONB uvw(direction);

  // Generates a random direction vector that points toward a sphere from a
  // point in space that is outside the sphere. This is used for a sphere to
  // sample on directions that actually intersect the sphere — this is faster
  // and more accurate.
  double r1 = random_double();
  double r2 = random_double();
  double z =
      1 + r2 * (std::sqrt(1 - m_radius * m_radius / distance_squared) - 1);

  double phi = 2 * PI * r1;
  double x = std::cos(phi) * std::sqrt(1 - z * z);
  double y = std::sin(phi) * std::sqrt(1 - z * z);

  return uvw.transform(Vec3(x, y, z));
}