#pragma once

class HittableList;  // From HittableList.hpp.
struct CameraConfig; // From CameraConfig.hpp.

void populate_cornell_box_scene(HittableList &world, HittableList &lights,
                                CameraConfig &camera_config);
void populate_bouncing_spheres_scene(HittableList &world, HittableList &lights,
                                     CameraConfig &camera_config);
