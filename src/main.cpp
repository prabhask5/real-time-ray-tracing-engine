#include "core/Hittable.hpp"
#include "core/camera/DynamicCamera.hpp"
#include "core/camera/StaticCamera.hpp"
#include "input/CLI.hpp"
#include "scene/Scene.hpp"
#include "scene/materials/DielectricMaterial.hpp"
#include "scene/materials/DiffuseLightMaterial.hpp"
#include "scene/materials/LambertianMaterial.hpp"
#include "scene/materials/MetalMaterial.hpp"
#include "scene/objects/Plane.hpp"
#include "scene/objects/PlaneUtility.hpp"
#include "scene/objects/RotateY.hpp"
#include "scene/objects/Sphere.hpp"
#include "scene/objects/Translate.hpp"
#include "scene/textures/CheckerTexture.hpp"
#include "utils/math/Vec3Utility.hpp"
#include <memory>

void populate_cornell_box_scene(HittableList &world, HittableList &lights,
                                CameraConfig &cam_config) {
  auto red = std::make_shared<LambertianMaterial>(Color(.65, .05, .05));
  auto white = std::make_shared<LambertianMaterial>(Color(.73, .73, .73));
  auto green = std::make_shared<LambertianMaterial>(Color(.12, .45, .15));
  auto light = std::make_shared<DiffuseLightMaterial>(Color(15, 15, 15));

  // Cornell box sides.
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

  // Light.
  world.add(std::make_shared<Plane>(Point3(213, 554, 227), Vec3(130, 0, 0),
                                    Vec3(0, 0, 105), light));

  // Box.
  HittablePtr box1 = make_box(Point3(0, 0, 0), Point3(165, 330, 165), white);
  box1 = std::make_shared<RotateY>(box1, 15);
  box1 = std::make_shared<Translate>(box1, Vec3(265, 0, 295));
  world.add(box1);

  // Glass Sphere.
  auto glass = std::make_shared<DielectricMaterial>(1.5);
  world.add(std::make_shared<Sphere>(Point3(190, 90, 190), 90, glass));

  // Light Sources.
  auto empty_material = MaterialPtr();
  lights.add(std::make_shared<Plane>(Point3(343, 554, 332), Vec3(-130, 0, 0),
                                     Vec3(0, 0, -105), empty_material));
  lights.add(
      std::make_shared<Sphere>(Point3(190, 90, 190), 90, empty_material));

  // Populate custom cam config.
  cam_config.aspect_ratio = 1.0;
  cam_config.background = Color(0, 0, 0);
  cam_config.vfov = 40;
  cam_config.lookfrom = Point3(278, 278, -800);
  cam_config.lookat = Point3(278, 278, 0);
  cam_config.vup = Vec3(0, 1, 0);
  cam_config.defocus_angle = 0;
}

void populate_bouncing_spheres_scene(HittableList &world, HittableList &lights,
                                     CameraConfig &cam_config) {
  auto checker = std::make_shared<CheckerTexture>(0.32, Color(.2, .3, .1),
                                                  Color(.9, .9, .9));
  world.add(
      std::make_shared<Sphere>(Point3(0, -1000, 0), 1000,
                               std::make_shared<LambertianMaterial>(checker)));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      auto choose_mat = random_double();
      Point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

      if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
        MaterialPtr sphere_material;

        if (choose_mat < 0.8) {
          // Diffuse.
          auto albedo = Color::random() * Color::random();
          sphere_material = std::make_shared<LambertianMaterial>(albedo);
          auto center2 = center + Vec3(0, random_double(0, .5), 0);
          world.add(
              std::make_shared<Sphere>(center, center2, 0.2, sphere_material));
        } else if (choose_mat < 0.95) {
          // Metal.
          auto albedo = Color::random(0.5, 1);
          auto fuzz = random_double(0, 0.5);
          sphere_material = std::make_shared<MetalMaterial>(albedo, fuzz);
          world.add(std::make_shared<Sphere>(center, 0.2, sphere_material));
        } else {
          // Glass.
          sphere_material = std::make_shared<DielectricMaterial>(1.5);
          world.add(std::make_shared<Sphere>(center, 0.2, sphere_material));
        }
      }
    }
  }

  auto material1 = std::make_shared<DielectricMaterial>(1.5);
  world.add(std::make_shared<Sphere>(Point3(0, 1, 0), 1.0, material1));

  auto material2 = std::make_shared<LambertianMaterial>(Color(0.4, 0.2, 0.1));
  world.add(std::make_shared<Sphere>(Point3(-4, 1, 0), 1.0, material2));

  auto material3 = std::make_shared<MetalMaterial>(Color(0.7, 0.6, 0.5), 0.0);
  world.add(std::make_shared<Sphere>(Point3(4, 1, 0), 1.0, material3));

  // Populate custom cam config.
  cam_config.aspect_ratio = 16.0 / 9.0;
  cam_config.background = Color(0.70, 0.80, 1.00);
  cam_config.vfov = 20;
  cam_config.lookfrom = Point3(13, 2, 3);
  cam_config.lookat = Point3(0, 0, 0);
  cam_config.vup = Vec3(0, 1, 0);
  cam_config.defocus_angle = 0.6;
  cam_config.focus_dist = 10.0;
}

int main(int argc, char **argv) {
  CLIOptions options = parse_cli(argc, argv);
  if (options.help) {
    print_help();
    return 0;
  } else if (options.any_errors) {
    return 0;
  }

  constexpr bool is_cuda_enabled =
#ifdef USE_CUDA
      true;
#else
      false;
#endif

  HittableList world;
  HittableList lights;
  CameraConfig cam_config = {
      .image_width = options.width,
      .samples_per_pixel = options.samples,
      .max_depth = options.depth,
      .use_parallelism = options.use_parallelism,
      .use_bvh = options.use_bvh,
      .use_gpu = is_cuda_enabled && options.use_gpu,
      .use_wavefront = options.use_wavefront,
  };

  populate_cornell_box_scene(world, lights, cam_config);
  // populate_bouncing_spheres_scene(world, lights, cam_config);

  if (options.use_static) {
    // Set up static camera.
    StaticCamera cam(cam_config, options.static_output_file);
    cam.render(world, lights);
  } else {
    // Set up dynamic camera.
    DynamicCamera cam(cam_config);
    cam.render(world, lights);
  }
}