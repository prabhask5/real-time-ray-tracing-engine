#pragma once

#include <Hittable.hpp>
#include <HittableTypes.hpp>
#include <Interval.hpp>

bool bbox_compare(const HittablePtr a, const HittablePtr b, int axis_index) {
  const Interval &a_axis_interval =
      a->get_bounding_box().get_axis_interval(axis_index);
  const Interval &b_axis_interval =
      b->get_bounding_box().get_axis_interval(axis_index);
  return a_axis_interval.min() < b_axis_interval.min();
}

bool bbox_x_compare(const HittablePtr a, const HittablePtr b) {
  return bbox_compare(a, b, 0);
}

bool bbox_y_compare(const HittablePtr a, const HittablePtr b) {
  return bbox_compare(a, b, 1);
}

bool bbox_z_compare(const HittablePtr a, const HittablePtr b) {
  return bbox_compare(a, b, 2);
}