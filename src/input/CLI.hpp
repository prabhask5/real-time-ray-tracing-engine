#pragma once

#include <string>

// Memory layout optimized for CLI option processing
struct alignas(16) CLIOptions {
  // Hot data: most frequently accessed options.

  int width = 600;   // Image width (frequently checked)
  int samples = 100; // Sample count (frequently checked)
  int depth = 50;    // Ray depth (frequently checked)

  // Warm data: boolean flags grouped together for cache efficiency.

  bool help = false;            // Help flag
  bool use_static = true;       // Static rendering mode
  bool use_parallelism = false; // Parallelism flag
  bool use_bvh = false;         // BVH acceleration flag
  bool use_gpu = false;         // GPU rendering flag
  bool any_errors = false;      // Error tracking flag

  // Cold data: string data (larger, less frequently accessed).

  std::string static_output_file = "image.ppm"; // Output filename
};

CLIOptions parse_cli(int argc, char **argv);

void print_help();