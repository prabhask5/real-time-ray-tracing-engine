#pragma once

#include "../scene/materials/Material.hpp"
#include "../scene/materials/MaterialTypes.hpp"
#include "../utils/math/Vec3.hpp"
#include "../utils/math/Vec3Utility.hpp"
#include "Ray.hpp"
#include "Vec3Types.hpp"
#include <iomanip>
#include <sstream>

// Represents captured info from a ray hitting a hittable object.
// Memory layout optimized for cache efficiency and SIMD compatibility.
struct alignas(16) HitRecord {
  // Hot data: most frequently accessed members first.

  // The point it hit at.
  Point3 point;

  // The normal vector to the ray/object intersection.
  Vec3 normal;

  // Group doubles together for optimal packing and cache usage.

  // The parameter t, time, along the ray in which the ray hit the hittable
  // object.
  double t;

  // 2D coordinates used to map a 2D texture image onto a 3D surface. When a ray
  // hits a surface (like a sphere or triangle), the u and v values tell the
  // renderer which part of the texture to apply at that point. This would be
  // able to determine what the texture of the object looks like at the position
  // the ray hit the object.
  double u, v;

  // Cold data: less frequently accessed members last.

  // The material of the hittable object.
  MaterialPtr material;

  // Front face = the ray hits the surface from outside (normal opposes the
  // ray). Back face = the ray hits from inside (normal faces same way as ray).
  bool frontFace;

  // Sets the hit record normal vector.
  // NOTE: the parameter `outward_normal` is assumed to have unit length.
  void set_face_normal(const Ray &ray, const Vec3 &outward_normal) {
    frontFace = dot_product(ray.direction(), outward_normal) < 0;
    normal = frontFace ? outward_normal : -outward_normal;
  }

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"HitRecord\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"point\":" << point.json() << ",";
    oss << "\"normal\":" << normal.json() << ",";
    oss << "\"t\":" << t << ",";
    oss << "\"u\":" << u << ",";
    oss << "\"v\":" << v << ",";
    oss << "\"material\":" << (material ? material->json() : "null") << ",";
    oss << "\"frontFace\":" << (frontFace ? "true" : "false");
    oss << "}";
    return oss.str();
  }
};