#pragma once

#include <string>

struct CLIOptions {
  bool help = false;
  bool use_static = true;
  std::string static_output_file = "image.ppm";
  bool use_parallelism = false;
  bool use_bvh = false;
  int width = 600;
  int samples = 100;
  int depth = 50;
  bool any_errors = false;
};

CLIOptions parse_cli(int argc, char **argv);

void print_help();