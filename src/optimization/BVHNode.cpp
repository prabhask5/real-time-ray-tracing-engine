#include "BVHNode.hpp"
#include "../core/HitRecord.hpp"
#include "../core/HittableList.hpp"
#include "AABBUtility.hpp"
#include <algorithm>

BVHNode::BVHNode(HittableList list)
    : BVHNode(list.get_objects(), 0, list.get_objects().size()) {}

BVHNode::BVHNode(std::vector<HittablePtr> &objects, size_t start, size_t end) {
  // We build the BVH bbox incrementally by expanding it to include the next
  // hittable object.
  m_bbox = EMPTY_AABB; // We start empty.

  for (int i = start; i < end; ++i)
    m_bbox = AABB(m_bbox, objects[i]->get_bounding_box());

  int axis = m_bbox.get_longest_axis();
  auto comp_func = (axis == 0)   ? bbox_x_compare
                   : (axis == 1) ? bbox_y_compare
                                 : bbox_z_compare;
  size_t object_span = end - start;

  if (object_span == 1)
    m_left = m_right = objects[start];
  else if (object_span == 2) {
    m_left = objects[start];
    m_right = objects[start + 1];
  } else {
    std::sort(std::begin(objects) + start, std::begin(objects) + end,
              comp_func);

    // We recursively make the children BVHNodes by splitting the objects list.
    size_t mid = start + object_span / 2;
    m_left = std::make_shared<BVHNode>(objects, start, mid);
    m_right = std::make_shared<BVHNode>(objects, mid, end);
  }
}

AABB BVHNode::get_bounding_box() const { return m_bbox; }

bool BVHNode::hit(const Ray &ray, Interval t_values, HitRecord &record) const {
  if (!m_bbox.hit(ray, t_values))
    return false;

  // Recursively check if the left and right child bbox's are hit.
  bool left_hit = m_left->hit(ray, t_values, record);
  bool right_hit = m_right->hit(
      ray, Interval(t_values.min(), left_hit ? record.t : t_values.max()),
      record);
  // NOTE: For above, the ternary is an optimization since we know the ray must
  // hit the left child before the right child, so we can only look at the slice
  // if so.

  return left_hit || right_hit;
}