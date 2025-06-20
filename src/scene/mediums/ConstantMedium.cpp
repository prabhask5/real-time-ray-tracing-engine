#include "ConstantMedium.hpp"
#include "../../core/HitRecord.hpp"
#include "../../optimization/AABB.hpp"
#include "../materials/IsotropicMaterial.hpp"

ConstantMedium::ConstantMedium(HittablePtr boundary, double density,
                               TexturePtr texture)
    : m_boundary(boundary), m_density(density),
      m_phase_function(std::make_shared<IsotropicMaterial>(texture)) {}

ConstantMedium::ConstantMedium(HittablePtr boundary, double density,
                               const Color &albedo)
    : m_boundary(boundary), m_density(density),
      m_phase_function(std::make_shared<IsotropicMaterial>(albedo)) {}

AABB ConstantMedium::get_bounding_box() const {
  return m_boundary->get_bounding_box();
}

bool ConstantMedium::hit(const Ray &ray, Interval t_values,
                         HitRecord &record) const {
  HitRecord rec1, rec2;

  // Find if the ray hits and enters the boundary.
  if (!m_boundary->hit(ray, UNIVERSE_INTERVAL, rec1))
    return false;

  // Find if the ray exits the boundary, only if the ray enters the boundary.
  if (!m_boundary->hit(ray, Interval(rec1.t + 0.0001, INF), rec2))
    return false;

  if (rec1.t < t_values.min())
    rec1.t = t_values.min();
  if (rec2.t > t_values.max())
    rec2.t = t_values.max();

  if (rec1.t >= rec2.t)
    return false;

  if (rec1.t < 0)
    rec1.t = 0;

  // This simulates a random scattering distance inside the medium using an
  // exponential distribution (common for modeling scattering).
  double ray_length = ray.direction().length();
  double distance_inside_boundary = (rec2.t - rec1.t) * ray_length;

  // This randomly determines when the ray scatters inside the medium.
  double neg_inv_density = -1.0 / m_density;
  double hit_distance = neg_inv_density * std::log(random_double());

  // If the scattering distance is longer than the ray distance inside the
  // medium boundary, it does not scatter.
  if (hit_distance > distance_inside_boundary)
    return false;

  record.t = rec1.t + hit_distance / ray_length;
  record.point = ray.at(record.t);

  record.normal = Vec3(1, 0, 0); // arbitrary
  record.frontFace = true;       // also arbitrary
  record.material = m_phase_function;

  return true;
}

HittablePtr ConstantMedium::get_boundary() const { return m_boundary; }

double ConstantMedium::get_density() const { return m_density; }

MaterialPtr ConstantMedium::get_phase_function() const {
  return m_phase_function;
}