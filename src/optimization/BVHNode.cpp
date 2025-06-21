#include "BVHNode.hpp"
#include "../core/HitRecord.hpp"
#include "../core/HittableList.hpp"
#include "../utils/math/Utility.hpp"
#include "AABBUtility.hpp"
#include <algorithm>
#include <execution>
#include <functional>
#include <future>
#include <limits>
#include <numeric>
#include <thread>

BVHNode::BVHNode(HittableList list)
    : BVHNode(list.get_objects(), 0, list.get_objects().size(),
              DEFAULT_MAX_LEAF_SIZE, true) {}

BVHNode::BVHNode(std::vector<HittablePtr> &objects, size_t start, size_t end)
    : BVHNode(objects, start, end, DEFAULT_MAX_LEAF_SIZE, true) {}

BVHNode::BVHNode(std::vector<HittablePtr> &objects, size_t start, size_t end,
                 int max_leaf_size, bool use_parallel) {
  // Build bounding box for all objects in range.
  m_bbox = EMPTY_AABB;
  for (size_t i = start; i < end; ++i) {
    m_bbox = AABB(m_bbox, objects[i]->get_bounding_box());
  }

  size_t object_span = end - start;

  // Create leaf node if we have few enough objects.
  if (object_span <= static_cast<size_t>(max_leaf_size)) {
    if (object_span == 1) {
      m_left = m_right = objects[start];
    } else if (object_span == 2) {
      m_left = objects[start];
      m_right = objects[start + 1];
    } else {
      // For small leaf nodes, just split in half.
      size_t mid = start + object_span / 2;
      m_left = std::make_shared<BVHNode>(objects, start, mid, max_leaf_size,
                                         use_parallel);
      m_right = std::make_shared<BVHNode>(objects, mid, end, max_leaf_size,
                                          use_parallel);
    }
    return;
  }

  // Build centroid bounding box for SAH.
  AABB centroid_bbox = EMPTY_AABB;
  for (size_t i = start; i < end; ++i) {
    AABB obj_bbox = objects[i]->get_bounding_box();
    Vec3 centroid = obj_bbox.center();
    centroid_bbox = AABB(centroid_bbox, AABB(centroid, centroid));
  }

  // Find best SAH split.
  SAHSplit best_split = find_best_sah_split(objects, start, end, centroid_bbox);

  // Fall back to spatial median if SAH doesn't find good split.
  if (best_split.cost == INF || best_split.left_count == 0 ||
      best_split.right_count == 0) {
    int axis = m_bbox.get_longest_axis();
    auto comp_func = (axis == 0)   ? bbox_x_compare
                     : (axis == 1) ? bbox_y_compare
                                   : bbox_z_compare;

    std::sort(std::begin(objects) + start, std::begin(objects) + end,
              comp_func);
    size_t mid = start + object_span / 2;

    m_left = std::make_shared<BVHNode>(objects, start, mid, max_leaf_size,
                                       use_parallel);
    m_right = std::make_shared<BVHNode>(objects, mid, end, max_leaf_size,
                                        use_parallel);
    return;
  }

  // Partition objects based on SAH split.
  size_t mid;
  if (use_parallel && object_span > 1000) {
    mid = parallel_partition(objects, start, end, best_split.axis,
                             best_split.position);
  } else {
    // Sequential partition.
    auto partition_point =
        std::partition(objects.begin() + start, objects.begin() + end,
                       [best_split](const HittablePtr &obj) {
                         AABB bbox = obj->get_bounding_box();
                         Vec3 centroid = bbox.center();
                         return centroid[best_split.axis] < best_split.position;
                       });
    mid = std::distance(objects.begin(), partition_point);
  }

  // Ensure we don't create empty partitions.
  if (mid == start || mid == end) {
    mid = start + object_span / 2;
  }

  // Recursively build children.
  if (use_parallel && object_span > 10000) {
    // Build children in parallel for large nodes.
    auto left_future = std::async(std::launch::async, [&]() {
      return std::make_shared<BVHNode>(objects, start, mid, max_leaf_size,
                                       use_parallel);
    });
    m_right = std::make_shared<BVHNode>(objects, mid, end, max_leaf_size,
                                        use_parallel);
    m_left = left_future.get();
  } else {
    m_left = std::make_shared<BVHNode>(objects, start, mid, max_leaf_size,
                                       use_parallel);
    m_right = std::make_shared<BVHNode>(objects, mid, end, max_leaf_size,
                                        use_parallel);
  }

  // Build flattened representation for efficient traversal.
  if (object_span > 100) { // Only flatten for larger scenes.
    build_flattened_tree();
    m_use_flattened = true;
  }
}

AABB BVHNode::get_bounding_box() const { return m_bbox; }

bool BVHNode::hit(const Ray &ray, Interval t_values, HitRecord &record) const {
  // Use flattened traversal for better cache performance if available.
  if (m_use_flattened) {
    return hit_flattened(ray, t_values, record);
  }

  // Original pointer-based traversal.
  if (!m_bbox.hit(ray, t_values))
    return false;

  // Recursively check if the left and right child bboxes are hit.
  bool left_hit = m_left->hit(ray, t_values, record);
  bool right_hit = m_right->hit(
      ray, Interval(t_values.min(), left_hit ? record.t : t_values.max()),
      record);
  // NOTE: For above, the ternary is an optimization since we know the ray must
  // hit the left child before the right child, so we can only look at the slice
  // if so.

  return left_hit || right_hit;
}

double BVHNode::pdf_value(const Point3 &origin, const Vec3 &direction) const {
  // For a BVH node, the pdf value biasing should be just the average of all the
  // PDF value biasing of the inner hittable objects. To do this, we can
  // recursively get the children BVH node PDF values and take the average of
  // both of those.

  return 0.5 * m_left->pdf_value(origin, direction) +
         0.5 * m_right->pdf_value(origin, direction);
}

Vec3 BVHNode::random(const Point3 &origin) const {
  // Randomly chooses one BVH node child and returns a direction vector
  // sampled from it, just random() recursively.

  if (random_int(0, 1) == 0)
    return m_left->random(origin);
  return m_right->random(origin);
}

SAHSplit BVHNode::find_best_sah_split(const std::vector<HittablePtr> &objects,
                                      size_t start, size_t end,
                                      const AABB &centroid_bbox) const {
  SAHSplit best_split;
  best_split.cost = INF;

  // Try each axis.
  for (int axis = 0; axis < 3; ++axis) {
    double axis_min = centroid_bbox.get_axis_interval(axis).min();
    double axis_max = centroid_bbox.get_axis_interval(axis).max();

    // Skip if all centroids are at same position on this axis.
    if (axis_max - axis_min < 1e-9)
      continue;

    // Sample split positions along axis.
    for (int i = 1; i < SAH_SAMPLES; ++i) {
      double t = static_cast<double>(i) / SAH_SAMPLES;
      double position = axis_min + t * (axis_max - axis_min);

      double cost = calculate_sah_cost(objects, start, end, axis, position);

      if (cost < best_split.cost) {
        best_split.axis = axis;
        best_split.position = position;
        best_split.cost = cost;

        // Count objects on each side for validation.
        size_t left_count = 0, right_count = 0;
        for (size_t j = start; j < end; ++j) {
          AABB bbox = objects[j]->get_bounding_box();
          Vec3 centroid = bbox.center();
          if (centroid[axis] < position) {
            ++left_count;
          } else {
            ++right_count;
          }
        }
        best_split.left_count = left_count;
        best_split.right_count = right_count;
      }
    }
  }

  return best_split;
}

double BVHNode::calculate_sah_cost(const std::vector<HittablePtr> &objects,
                                   size_t start, size_t end, int axis,
                                   double position) const {
  AABB left_bbox = EMPTY_AABB;
  AABB right_bbox = EMPTY_AABB;
  size_t left_count = 0, right_count = 0;

  // Count objects and build bounding boxes for each side.
  for (size_t i = start; i < end; ++i) {
    AABB bbox = objects[i]->get_bounding_box();
    Vec3 centroid = bbox.center();

    if (centroid[axis] < position) {
      left_bbox = AABB(left_bbox, bbox);
      ++left_count;
    } else {
      right_bbox = AABB(right_bbox, bbox);
      ++right_count;
    }
  }

  // Avoid degenerate splits.
  if (left_count == 0 || right_count == 0) {
    return INF;
  }

  // Calculate SAH cost: C_traversal + P_left * C_left + P_right * C_right.
  double total_surface_area = m_bbox.surface_area();
  if (total_surface_area < 1e-9)
    return INF;

  double left_surface_area = left_bbox.surface_area();
  double right_surface_area = right_bbox.surface_area();

  double p_left = left_surface_area / total_surface_area;
  double p_right = right_surface_area / total_surface_area;

  return TRAVERSAL_COST + p_left * left_count * INTERSECTION_COST +
         p_right * right_count * INTERSECTION_COST;
}

size_t BVHNode::parallel_partition(std::vector<HittablePtr> &objects,
                                   size_t start, size_t end, int axis,
                                   double position) const {
  // For very large arrays, use parallel partitioning.
  size_t num_threads = std::thread::hardware_concurrency();
  size_t chunk_size = (end - start) / num_threads;

  if (chunk_size < 100) {
    // Fall back to sequential for small chunks.
    auto partition_point =
        std::partition(objects.begin() + start, objects.begin() + end,
                       [axis, position](const HittablePtr &obj) {
                         AABB bbox = obj->get_bounding_box();
                         Vec3 centroid = bbox.center();
                         return centroid[axis] < position;
                       });
    return std::distance(objects.begin(), partition_point);
  }

  // Multi-threaded approach: classify objects in parallel, then rearrange.
  std::vector<bool> is_left(end - start);
  std::vector<std::future<void>> futures;

  // Phase 1: Classify objects in parallel.
  for (size_t t = 0; t < num_threads; ++t) {
    size_t chunk_start = start + t * chunk_size;
    size_t chunk_end = (t == num_threads - 1) ? end : chunk_start + chunk_size;

    if (chunk_start >= chunk_end)
      break;

    futures.emplace_back(std::async(
        std::launch::async, [&, chunk_start, chunk_end, axis, position]() {
          for (size_t i = chunk_start; i < chunk_end; ++i) {
            AABB bbox = objects[i]->get_bounding_box();
            Vec3 centroid = bbox.center();
            is_left[i - start] = centroid[axis] < position;
          }
        }));
  }

  // Wait for classification to complete.
  for (auto &future : futures) {
    future.wait();
  }

  // Phase 2: Count left objects and rearrange.
  size_t left_count = std::count(is_left.begin(), is_left.end(), true);

  std::vector<HittablePtr> temp_objects(end - start);
  size_t left_idx = 0, right_idx = left_count;

  for (size_t i = 0; i < end - start; ++i) {
    if (is_left[i]) {
      temp_objects[left_idx++] = objects[start + i];
    } else {
      temp_objects[right_idx++] = objects[start + i];
    }
  }

  // Copy back to original array.
  std::copy(temp_objects.begin(), temp_objects.end(), objects.begin() + start);

  return start + left_count;
}

void BVHNode::build_flattened_tree() {
  m_flat_nodes.clear();
  m_primitives.clear();

  // Reserve space for efficiency.
  m_flat_nodes.reserve(1000);
  m_primitives.reserve(100);

  uint32_t offset = 0;
  flatten_recursive(std::make_shared<BVHNode>(*this), offset);
}

uint32_t BVHNode::flatten_recursive(const HittablePtr &node,
                                    uint32_t &node_offset) {
  uint32_t my_offset = node_offset++;

  // Ensure we have space.
  if (my_offset >= m_flat_nodes.size()) {
    m_flat_nodes.resize(my_offset + 1);
  }

  FlatNode &flat_node = m_flat_nodes[my_offset];
  flat_node.bbox = node->get_bounding_box();

  // Check if this is a BVH node.
  std::shared_ptr<BVHNode> bvh_node = std::dynamic_pointer_cast<BVHNode>(node);
  if (bvh_node && bvh_node->m_left && bvh_node->m_right &&
      bvh_node->m_left != bvh_node->m_right) {
    // Internal node
    flat_node.is_leaf = 0;

    // Recursively flatten children.
    // Left child immediately follows parent.
    flat_node.inner.left_offset = node_offset;
    flatten_recursive(bvh_node->m_left, node_offset);

    // Right child follows all left subtree nodes.
    flat_node.inner.right_offset = node_offset;
    flatten_recursive(bvh_node->m_right, node_offset);
  } else {
    // Leaf node - store primitives.
    flat_node.is_leaf = 1;
    flat_node.leaf.prim_offset = static_cast<uint32_t>(m_primitives.size());

    if (bvh_node && bvh_node->m_left == bvh_node->m_right) {
      // Single primitive.
      m_primitives.push_back(bvh_node->m_left);
      flat_node.leaf.prim_count = 1;
    } else if (bvh_node) {
      // Two primitives.
      m_primitives.push_back(bvh_node->m_left);
      m_primitives.push_back(bvh_node->m_right);
      flat_node.leaf.prim_count = 2;
    } else {
      // Non-BVH node, just store it.
      m_primitives.push_back(node);
      flat_node.leaf.prim_count = 1;
    }
  }

  return my_offset;
}

bool BVHNode::hit_flattened(const Ray &ray, Interval t_values,
                            HitRecord &record) const {
  if (m_flat_nodes.empty())
    return false;

  bool hit_anything = false;

  // Stack for iterative traversal (avoids recursion overhead).
  struct StackEntry {
    uint32_t node_idx;
    double t_min;
  };

  StackEntry stack[64];
  int stack_ptr = 0;

  // Start with root.
  stack[stack_ptr++] = {0, t_values.min()};

  while (stack_ptr > 0) {
    StackEntry entry = stack[--stack_ptr];

    // Skip if we've found a closer hit.
    if (entry.t_min >= t_values.max())
      continue;

    const FlatNode &node = m_flat_nodes[entry.node_idx];

    // Test against node's bounding box.
    Interval node_t = t_values;
    if (!node.bbox.hit(ray, node_t))
      continue;

    if (node.is_leaf) {
      // Test primitives.
      for (uint32_t i = 0; i < node.leaf.prim_count; ++i) {
        uint32_t prim_idx = node.leaf.prim_offset + i;
        if (m_primitives[prim_idx]->hit(ray, t_values, record)) {
          hit_anything = true;
          t_values.set_interval(t_values.min(), record.t);
        }
      }
    } else {
      // Order children based on ray direction for better performance.
      uint32_t first = node.inner.left_offset;
      uint32_t second = node.inner.right_offset;

      // Simple ordering heuristic.
      if (ray.direction()[node.bbox.get_longest_axis()] < 0) {
        std::swap(first, second);
      }

      // Push children (far first, near second so near is processed first).
      if (stack_ptr < 63) {
        stack[stack_ptr++] = {second, node_t.min()};
        stack[stack_ptr++] = {first, node_t.min()};
      }
    }
  }

  return hit_anything;
}
