#include "core/HittableList.hpp"
#include "core/StaticCamera.hpp"
#include "scene/BVHNode.hpp"
#include "scene/DielectricMaterial.hpp"
#include "scene/LambertianMaterial.hpp"
#include "scene/MetalMaterial.hpp"
#include "scene/Sphere.hpp"
#include <memory>

int main() {
  HittableList world;

  // Make the ground diffuse.
  auto ground_material =
      std::make_shared<LambertianMaterial>(Color(0.5, 0.5, 0.5));
  world.add(
      std::make_shared<Sphere>(Point3(0, -1000, 0), 1000, ground_material));

  // Add spheres.

  auto material_one = std::make_shared<DielectricMaterial>(1.5);
  world.add(std::make_shared<Sphere>(Point3(0, 1, 0), 1.0, material_one));

  auto material_two =
      std::make_shared<LambertianMaterial>(Color(0.4, 0.2, 0.1));
  world.add(std::make_shared<Sphere>(Point3(-4, 1, 0), 1.0, material_two));

  auto material_three =
      std::make_shared<MetalMaterial>(Color(0.7, 0.6, 0.5), 0.0);
  world.add(std::make_shared<Sphere>(Point3(4, 1, 0), 1.0, material_three));

  // Build BVH for acceleration.
  BVHNode bvh(world.objects());

  // Set up static camera.

  StaticCameraConfig cam_config = {.aspect_ratio = 16.0 / 9.0,
                                   .image_width = 1200,
                                   .samples_per_pixel = 10,
                                   .max_depth = 20,

                                   .vfov = 20,
                                   .lookfrom = Point3(13, 2, 3),
                                   .lookat = Point3(0, 0, 0),
                                   .vup = Vec3(0, 1, 0),

                                   .defocus_angle = 0.6,
                                   .focus_dist = 10.0};

  StaticCamera cam(cam_config);

  cam.render(bvh);
}