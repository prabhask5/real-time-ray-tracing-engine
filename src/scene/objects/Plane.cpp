#include "Plane.hpp"
#include "../../core/HitRecord.hpp"
#include "../../core/Ray.hpp"
#include "../../utils/math/Vec3Utility.hpp"

Plane::Plane(const Point3 &corner, const Vec3 &u_side, const Vec3 &v_side,
             MaterialPtr material)
    : m_corner(corner), m_u_side(u_side), m_v_side(v_side),
      m_material(material) {
  Vec3 n = cross_product(u_side, v_side);

  m_normal = unit_vector(n);
  m_D = dot_product(m_normal, corner);
  m_w = n / dot_product(n, n);
  m_surface_area = n.length();

  // Compute the bounding box of all four vertices.
  AABB bbox_diagonal_one = AABB(corner, corner + u_side + v_side);
  AABB bbox_diagonal_two = AABB(corner + u_side, corner + v_side);
  m_bbox = AABB(bbox_diagonal_one, bbox_diagonal_two);
}

AABB Plane::get_bounding_box() const { return m_bbox; }

bool Plane::hit(const Ray &ray, Interval t_values, HitRecord &record) const {
  double denom = dot_product(m_normal, ray.direction());

  // No hit if the ray is parallel to the plane.
  if (std::fabs(denom) < 1e-8)
    return false;

  // Return false if the hit point parameter t is outside the ray interval.
  double t = (m_D - dot_product(m_normal, ray.origin())) / denom;
  if (!t_values.contains(t))
    return false;

  // Determine if the hit point lies within the planar shape using its plane
  // coordinates.
  Point3 intersection = ray.at(t);
  Vec3 planar_hitpt_vector = intersection - m_corner;
  double alpha = dot_product(m_w, cross_product(planar_hitpt_vector, m_v_side));
  double beta = dot_product(m_w, cross_product(m_u_side, planar_hitpt_vector));

  Interval unit_interval = Interval(0, 1);
  // Given the hit point in plane coordinates, return false if it is outside the
  // primitive, otherwise set the hit record UV coordinates and return true.

  if (!unit_interval.contains(alpha) || !unit_interval.contains(beta))
    return false;

  record.u = alpha;
  record.v = beta;

  // Ray hits the 2D shape; set the rest of the hit record and return true.
  record.t = t;
  record.point = intersection;
  record.material = m_material;
  record.set_face_normal(ray, m_normal);

  return true;
}

double Plane::pdf_value(const Point3 &origin, const Vec3 &direction) const {
  HitRecord record;
  if (!this->hit(Ray(origin, direction), Interval(0.001, INF), record))
    return 0;

  double distance_squared = record.t * record.t * direction.length_squared();
  double cosine =
      std::fabs(dot_product(direction, record.normal) / direction.length());

  // PDF formula for solid angle sampling.
  return distance_squared / (cosine * m_surface_area);
}

Vec3 Plane::random(const Point3 &origin) const {
  // Pick a random point on the rectangle.
  Point3 p =
      m_corner + (random_double() * m_u_side) + (random_double() * m_v_side);
  return p - origin;
}