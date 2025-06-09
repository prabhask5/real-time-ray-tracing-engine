#pragma once

#include "../../core/HittableList.hpp"
#include "Plane.hpp"
#include <MaterialTypes.hpp>

// Returns the 3D box (six sides) that contains the two opposite vertices a & b.
std::shared_ptr<HittableList> make_box(const Point3 &a, const Point3 &b,
                                       MaterialPtr material) {
  std::shared_ptr<HittableList> sides = std::make_shared<HittableList>();

  // Construct the two opposite vertices with the minimum and maximum
  // coordinates.
  Point3 min = Point3(std::fmin(a.x(), b.x()), std::fmin(a.y(), b.y()),
                      std::fmin(a.z(), b.z()));
  Point3 max = Point3(std::fmax(a.x(), b.x()), std::fmax(a.y(), b.y()),
                      std::fmax(a.z(), b.z()));

  Vec3 dx = Vec3(max.x() - min.x(), 0, 0);
  Vec3 dy = Vec3(0, max.y() - min.y(), 0);
  Vec3 dz = Vec3(0, 0, max.z() - min.z());

  sides->add(std::make_shared<Plane>(Point3(min.x(), min.y(), max.z()), dx, dy,
                                     material)); // front
  sides->add(std::make_shared<Plane>(Point3(max.x(), min.y(), max.z()), -dz, dy,
                                     material)); // right
  sides->add(std::make_shared<Plane>(Point3(max.x(), min.y(), min.z()), -dx, dy,
                                     material)); // back
  sides->add(std::make_shared<Plane>(Point3(min.x(), min.y(), min.z()), dz, dy,
                                     material)); // left
  sides->add(std::make_shared<Plane>(Point3(min.x(), max.y(), max.z()), dx, -dz,
                                     material)); // top
  sides->add(std::make_shared<Plane>(Point3(min.x(), min.y(), min.z()), dx, dz,
                                     material)); // bottom

  return sides;
}