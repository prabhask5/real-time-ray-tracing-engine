#include "CLI.hpp"
#include <iostream>

CLIOptions parse_cli(int argc, char **argv) {
  CLIOptions opts;
  bool output_selected = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      opts.help = true;
    } else if (arg == "--camera") {
      if (i + 1 < argc) {
        std::string type = argv[++i];
        if (type == "static") {
          opts.use_static = true;
        } else if (type == "dynamic") {
          opts.use_static = false;
        } else {
          opts.any_errors = true;
          std::cerr << "Unknown camera type: " << type << std::endl;
        }
      } else {
        opts.any_errors = true;
        std::cerr << "--camera requires an argument: static or dynamic\n";
      }
    } else if (arg == "--output") {
      if (i + 1 < argc) {
        opts.static_output_file = argv[++i];
        output_selected = true;
      } else {
        opts.any_errors = true;
        std::cerr << "--output requires a filename\n";
      }
    } else if (arg == "-p" || arg == "--parallel") {
      opts.use_parallelism = true;
    } else if (arg == "-b" || arg == "--bvh") {
      opts.use_bvh = true;
    } else if (arg == "--width") {
      if (i + 1 < argc) {
        try {
          opts.width = std::stoi(argv[++i]);
        } catch (...) {
          opts.any_errors = true;
          std::cerr << "--width requires a valid integer\n";
        }
      } else {
        opts.any_errors = true;
        std::cerr << "--width requires a number\n";
      }
    } else if (arg == "--samples") {
      if (i + 1 < argc) {
        try {
          opts.samples = std::stoi(argv[++i]);
        } catch (...) {
          opts.any_errors = true;
          std::cerr << "--samples requires a valid integer\n";
        }
      } else {
        opts.any_errors = true;
        std::cerr << "--samples requires a number\n";
      }
    } else if (arg == "--depth") {
      if (i + 1 < argc) {
        try {
          opts.depth = std::stoi(argv[++i]);
        } catch (...) {
          opts.any_errors = true;
          std::cerr << "--depth requires a valid integer\n";
        }
      } else {
        opts.any_errors = true;
        std::cerr << "--depth requires a number\n";
      }
    } else {
      opts.any_errors = true;
      std::cerr << "Unknown option: " << arg << std::endl;
    }
  }

  // Only allow setting an output file if camera is static.
  if (!opts.use_static && output_selected)
    std::cerr << "You can only set an output file if the static camera is "
                 "selected, ignoring...\n";

  return opts;
}

void print_help() {
  std::cout << "Raytracer: A high-performance real-time ray tracing engine.\n";
  std::cout << "Renders scenes using either a static or dynamic camera and "
               "supports parallelism and acceleration structures.\n\n";
  std::cout << "Usage: raytracer [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  -h, --help                 Show this help message\n";
  std::cout
      << "  --camera [static|dynamic]  Select camera type (default: static)\n";
  std::cout << "  --output <file>            Output file name for static "
               "camera (default: image.ppm)\n";
  std::cout << "  -p, --parallel             Enable multithreaded rendering\n";
  std::cout << "  -b, --bvh                  Use bounding volume hierarchy for "
               "scene acceleration\n";
  std::cout << "  --width <int>              Image width for scene render "
               "(default: 600)\n";
  std::cout << "  --samples <int>            Number of samples per pixel for "
               "ray tracing- higher number reduces fuzz, but increases "
               "rendering time (default: 10)\n";
  std::cout << "  --depth <int>              Number of recursive light bounces "
               "to track- higher number makes ray tracing better, but "
               "increases rendering time (default: 10)\n";
  std::cout << "\nExamples:\n";
  std::cout
      << "  raytracer --camera static --output render.ppm --parallel --bvh\n";
  std::cout << "  raytracer --camera dynamic --parallel\n";
}
