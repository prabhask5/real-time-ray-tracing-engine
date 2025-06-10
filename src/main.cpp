#include "core/HittableList.hpp"
#include "core/camera/StaticCamera.hpp"
#include "optimization/BVHNode.hpp"
#include "scene/materials/DielectricMaterial.hpp"
#include "scene/materials/DiffuseLightMaterial.hpp"
#include "scene/materials/LambertianMaterial.hpp"
#include "scene/materials/MetalMaterial.hpp"
#include "scene/objects/Plane.hpp"
#include "scene/objects/PlaneUtility.hpp"
#include "scene/objects/RotateY.hpp"
#include "scene/objects/Sphere.hpp"
#include "scene/objects/Translate.hpp"
#include <Hittable.hpp>
#include <memory>

int main() {
  HittableList world;

  auto red = std::make_shared<LambertianMaterial>(Color(.65, .05, .05));
  auto white = std::make_shared<LambertianMaterial>(Color(.73, .73, .73));
  auto green = std::make_shared<LambertianMaterial>(Color(.12, .45, .15));
  auto light = std::make_shared<DiffuseLightMaterial>(Color(15, 15, 15));

  // Cornell box sides
  world.add(std::make_shared<Plane>(Point3(555, 0, 0), Vec3(0, 0, 555),
                                    Vec3(0, 555, 0), green));
  world.add(std::make_shared<Plane>(Point3(0, 0, 555), Vec3(0, 0, -555),
                                    Vec3(0, 555, 0), red));
  world.add(std::make_shared<Plane>(Point3(0, 555, 0), Vec3(555, 0, 0),
                                    Vec3(0, 0, 555), white));
  world.add(std::make_shared<Plane>(Point3(0, 0, 555), Vec3(555, 0, 0),
                                    Vec3(0, 0, -555), white));
  world.add(std::make_shared<Plane>(Point3(555, 0, 555), Vec3(-555, 0, 0),
                                    Vec3(0, 555, 0), white));

  // Light
  world.add(std::make_shared<Plane>(Point3(213, 554, 227), Vec3(130, 0, 0),
                                    Vec3(0, 0, 105), light));

  // Box
  std::shared_ptr<Hittable> box1 =
      make_box(Point3(0, 0, 0), Point3(165, 330, 165), white);
  box1 = std::make_shared<RotateY>(box1, 15);
  box1 = std::make_shared<Translate>(box1, Vec3(265, 0, 295));
  world.add(box1);

  // Glass Sphere
  auto glass = std::make_shared<DielectricMaterial>(1.5);
  world.add(std::make_shared<Sphere>(Point3(190, 90, 190), 90, glass));

  // Light Sources
  auto empty_material = std::shared_ptr<Material>();
  HittableList lights;
  lights.add(std::make_shared<Plane>(Point3(343, 554, 332), Vec3(-130, 0, 0),
                                     Vec3(0, 0, -105), empty_material));
  lights.add(
      std::make_shared<Sphere>(Point3(190, 90, 190), 90, empty_material));

  // Set up static camera.

  CameraConfig cam_config = {.aspect_ratio = 1.0,
                             .image_width = 600,
                             .samples_per_pixel = 100,
                             .max_depth = 50,
                             .background = Color(0, 0, 0),

                             .vfov = 40,
                             .lookfrom = Point3(278, 278, -800),
                             .lookat = Point3(278, 278, 0),
                             .vup = Vec3(0, 1, 0),

                             .defocus_angle = 0};

  StaticCamera cam(cam_config, "image.ppm");

  // Use BVH Nodes for optimization.
  world = HittableList(std::make_shared<BVHNode>(world));
  lights = HittableList(std::make_shared<BVHNode>(lights));

  cam.render(world, lights);
}