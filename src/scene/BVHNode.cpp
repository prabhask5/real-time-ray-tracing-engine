#include "BVHNode.hpp"
#include "../core/HitRecord.hpp"
#include <Utility.hpp>
#include <algorithm>
#include <cstdlib>

namespace {

bool box_compare(const HittablePtr &a, const HittablePtr &b, int axis) {
  AABB box_a;
  AABB box_b;

  if (!a->bounding_box(box_a) || !b->bounding_box(box_b))
    return false;
  return box_a.min()[axis] < box_b.min()[axis];
}

bool box_x_compare(const HittablePtr &a, const HittablePtr &b) {
  return box_compare(a, b, 0);
}

bool box_y_compare(const HittablePtr &a, const HittablePtr &b) {
  return box_compare(a, b, 1);
}

bool box_z_compare(const HittablePtr &a, const HittablePtr &b) {
  return box_compare(a, b, 2);
}
} // namespace

BVHNode::BVHNode(const std::vector<HittablePtr> &objects, size_t start,
                 size_t end) {
  auto objs = objects;
  int axis = random_int(0, 2);
  auto comparator = (axis == 0)   ? box_x_compare
                    : (axis == 1) ? box_y_compare
                                  : box_z_compare;
  size_t object_span = end - start;

  if (object_span == 1) {
    m_left = m_right = objs[start];
  } else if (object_span == 2) {
    if (comparator(objs[start], objs[start + 1])) {
      m_left = objs[start];
      m_right = objs[start + 1];
    } else {
      m_left = objs[start + 1];
      m_right = objs[start];
    }
  } else {
    std::sort(objs.begin() + start, objs.begin() + end, comparator);
    size_t mid = start + object_span / 2;
    m_left = std::make_shared<BVHNode>(objs, start, mid);
    m_right = std::make_shared<BVHNode>(objs, mid, end);
  }

  AABB box_left, box_right;
  m_left->bounding_box(box_left);
  m_right->bounding_box(box_right);
  m_box = AABB::surrounding_box(box_left, box_right);
}

BVHNode::BVHNode(const std::vector<HittablePtr> &objects)
    : BVHNode(objects, 0, objects.size()) {}

bool BVHNode::hit(const Ray &ray, Interval t_values, HitRecord &record) const {
  if (!m_box.hit(ray, t_values))
    return false;

  bool hit_left = m_left->hit(ray, t_values, record);
  bool hit_right = m_right->hit(
      ray, Interval(t_values.min(), hit_left ? record.t : t_values.max()),
      record);
  return hit_left || hit_right;
}

bool BVHNode::bounding_box(AABB &output_box) const {
  output_box = m_box;
  return true;
}
