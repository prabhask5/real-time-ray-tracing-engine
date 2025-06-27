#pragma once

#include <iomanip>
#include <sstream>
#include <string>

// Memory layout optimized for CLI option processing.
struct alignas(16) CLIOptions {
  // Hot data: most frequently accessed options.

  int width = 600;   // Image width (frequently checked).
  int samples = 100; // Sample count (frequently checked).
  int depth = 50;    // Ray depth (frequently checked).

  // Warm data: boolean flags grouped together for cache efficiency.

  bool help = false;            // Help flag.
  bool use_static = true;       // Static rendering mode.
  bool use_parallelism = false; // Parallelism flag.
  bool use_bvh = false;         // BVH acceleration flag.
  bool use_gpu = false;         // GPU rendering flag.
  bool any_errors = false;      // Error tracking flag.
  bool debug = false;           // Debug mode flag.

  // Cold data: string data (larger, less frequently accessed).

  std::string static_output_file = "image.ppm"; // Output filename.

  // JSON serialization method.
  std::string json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{";
    oss << "\"type\":\"CLIOptions\",";
    oss << "\"address\":\"" << this << "\",";
    oss << "\"width\":" << width << ",";
    oss << "\"samples\":" << samples << ",";
    oss << "\"depth\":" << depth << ",";
    oss << "\"help\":" << (help ? "true" : "false") << ",";
    oss << "\"use_static\":" << (use_static ? "true" : "false") << ",";
    oss << "\"use_parallelism\":" << (use_parallelism ? "true" : "false")
        << ",";
    oss << "\"use_bvh\":" << (use_bvh ? "true" : "false") << ",";
    oss << "\"use_gpu\":" << (use_gpu ? "true" : "false") << ",";
    oss << "\"any_errors\":" << (any_errors ? "true" : "false") << ",";
    oss << "\"debug\":" << (debug ? "true" : "false") << ",";
    oss << "\"static_output_file\":\"" << static_output_file << "\"";
    oss << "}";
    return oss.str();
  }
};

CLIOptions parse_cli(int argc, char **argv);

void print_help();