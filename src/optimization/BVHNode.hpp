#pragma once

#include "AABB.hpp"
#include <Hittable.hpp>
#include <HittableTypes.hpp>
#include <vector>

class HittableList; // From HittableList.hpp.
class AABB;         // From AABB.hpp.

// Represents a bounding box that can contain multiple inner bounding boxes as
// children. A leaf node contains 1 or a few geometric objects. Optimizes the
// ray hit algorithm by ignoring all the inner bounding boxes in which the ray
// doesn't interact with the enclosing bounding box.
class BVHNode : public Hittable {
public:
  BVHNode(HittableList list);

  BVHNode(std::vector<HittablePtr> &objects, size_t start, size_t end);

  // Getter const methods.

  AABB get_bounding_box() const override;

  // Action methods.

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

private:
  HittablePtr m_left;
  HittablePtr m_right;
  AABB m_bbox;
};