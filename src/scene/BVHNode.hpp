#pragma once

#include "../core/AABB.hpp"
#include <Hittable.hpp>
#include <HittableTypes.hpp>
#include <vector>

class BVHNode : public Hittable {
public:
  BVHNode() = default;
  BVHNode(const std::vector<HittablePtr> &objects, size_t start, size_t end);
  BVHNode(const std::vector<HittablePtr> &objects);

  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;

  bool bounding_box(AABB &output_box) const override;

private:
  HittablePtr m_left;
  HittablePtr m_right;
  AABB m_box;
};
