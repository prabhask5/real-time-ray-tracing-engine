#pragma once

#include "Ray.hpp"
#include <MaterialTypes.hpp>
#include <Vec3.hpp>
#include <Vec3Types.hpp>
#include <Vec3Utility.hpp>

class Material;

// Represents captured info from a ray hitting a hittable object.
struct HitRecord {
  // The point it hit at.
  Point3 point;

  // The normal vector to the ray/object intersection.
  Vec3 normal;

  // The material of the hittable object.
  MaterialPtr material;

  // The parameter t, time, along the ray in which the ray hit the hittable
  // object.
  double t;

  // Front face = the ray hits the surface from outside (normal opposes the
  // ray). Back face = the ray hits from inside (normal faces same way as ray).
  bool frontFace;

  // Sets the hit record normal vector.
  // NOTE: the parameter `outward_normal` is assumed to have unit length.
  void set_face_normal(const Ray &ray, const Vec3 &outward_normal) {
    frontFace = dot_product(ray.direction(), outward_normal) < 0;
    normal = frontFace ? outward_normal : -outward_normal;
  }
};