#pragma once

#include "../core/Hittable.hpp"
#include "../core/HittableTypes.hpp"
#include "AABB.hpp"
#include <array>
#include <execution>
#include <iomanip>
#include <sstream>
#include <vector>

class HittableList; // From HittableList.hpp.
class AABB;         // From AABB.hpp.

// Represents a bounding box that can contain multiple inner bounding boxes as
// children. A leaf node contains 1 or a few geometric objects. Optimizes the
// ray hit algorithm by ignoring all the inner bounding boxes in which the ray
// doesn't interact with the enclosing bounding box.
// Memory layout optimized for BVH traversal performance.
// SAH split candidate for BVH construction.
struct SAHSplit {
  int axis;
  double position;
  double cost;
  size_t left_count;
  size_t right_count;

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"SAHSplit\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"axis\":" << axis << ",";
    oss << "\"position\":" << position << ",";
    oss << "\"cost\":" << cost << ",";
    oss << "\"left_count\":" << left_count << ",";
    oss << "\"right_count\":" << right_count;
    oss << "}";
    return oss.str();
  }
};

// Flattened BVH node for cache-efficient traversal,
struct alignas(32) FlatNode {
  AABB bbox;
  union {
    struct {
      uint32_t left_offset;  // Offset to left child
      uint32_t right_offset; // Offset to right child
    } inner;
    struct {
      uint32_t prim_offset; // Offset into primitive array
      uint32_t prim_count;  // Number of primitives
    } leaf;
  };
  uint8_t is_leaf;
  uint8_t padding[3];

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"FlatNode\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"bbox\":" << bbox.json() << ",";
    oss << "\"is_leaf\":" << (is_leaf ? "true" : "false") << ",";
    if (is_leaf) {
      oss << "\"prim_offset\":" << leaf.prim_offset << ",";
      oss << "\"prim_count\":" << leaf.prim_count;
    } else {
      oss << "\"left_offset\":" << inner.left_offset << ",";
      oss << "\"right_offset\":" << inner.right_offset;
    }
    oss << "}";
    return oss.str();
  }
};

class alignas(16) BVHNode : public Hittable {
public:
  BVHNode(HittableList list);
  BVHNode(std::vector<HittablePtr> &objects, size_t start, size_t end);

  // SAH-based construction with parallel partitioning.
  BVHNode(std::vector<HittablePtr> &objects, size_t start, size_t end,
          int max_leaf_size, bool use_parallel = true);

  // Direct constructor for conversion purposes
  BVHNode(HittablePtr left, HittablePtr right) : m_left(left), m_right(right) {
    if (left && right) {
      m_bbox = AABB(left->get_bounding_box(), right->get_bounding_box());
    } else if (left) {
      m_bbox = left->get_bounding_box();
    } else if (right) {
      m_bbox = right->get_bounding_box();
    }
  }

  // Getter const methods.
  AABB get_bounding_box() const override;
  HittablePtr get_left() const { return m_left; }
  HittablePtr get_right() const { return m_right; }
  bool is_leaf() const { return !m_left && !m_right; }

  // Action methods.
  bool hit(const Ray &ray, Interval t_values, HitRecord &record) const override;
  double pdf_value(const Point3 &origin, const Vec3 &direction) const override;
  Vec3 random(const Point3 &origin) const override;

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"BVHNode\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"bbox\":" << m_bbox.json() << ",";
    oss << "\"left\":" << (m_left ? m_left->json() : "null") << ",";
    oss << "\"right\":" << (m_right ? m_right->json() : "null") << ",";
    oss << "\"is_leaf\":" << (is_leaf() ? "true" : "false") << ",";
    oss << "\"use_flattened\":" << (m_use_flattened ? "true" : "false") << ",";
    oss << "\"flat_nodes_count\":" << m_flat_nodes.size() << ",";
    oss << "\"primitives_count\":" << m_primitives.size();
    oss << "}";
    return oss.str();
  }

private:
  // Build flattened representation after construction.
  void build_flattened_tree();
  uint32_t flatten_recursive(const HittablePtr &node, uint32_t &node_offset);

  // Optimized traversal using flattened nodes.
  bool hit_flattened(const Ray &ray, Interval t_values,
                     HitRecord &record) const;

private:
  // SAH cost calculation and split finding.
  SAHSplit find_best_sah_split(const std::vector<HittablePtr> &objects,
                               size_t start, size_t end,
                               const AABB &centroid_bbox) const;

  double calculate_sah_cost(const std::vector<HittablePtr> &objects,
                            size_t start, size_t end, int axis,
                            double position) const;

  // Parallel partitioning
  size_t parallel_partition(std::vector<HittablePtr> &objects, size_t start,
                            size_t end, int axis, double position) const;

  // Hot data: bounding box checked first in every traversal.
  AABB m_bbox;

  // Child pointers: accessed only if bbox test passes.
  HittablePtr m_left;
  HittablePtr m_right;

  // Flattened representation for cache-efficient traversal
  std::vector<FlatNode> m_flat_nodes;
  std::vector<HittablePtr> m_primitives;
  bool m_use_flattened = false;

  // Construction parameters.
  static constexpr int DEFAULT_MAX_LEAF_SIZE = 4;
  static constexpr int SAH_SAMPLES = 16;
  static constexpr double TRAVERSAL_COST = 1.0;
  static constexpr double INTERSECTION_COST = 2.0;
};