#pragma once

#include "Camera.hpp"
#include "CameraConfig.hpp"
#include <string>

class StaticCamera : public Camera {
public:
  StaticCamera(const CameraConfig &config, std::string output_file_name);

  // The main rendering loop:
  // - Calls initialize() to set up internal camera vectors.
  // - Outputs a header for the PPM image format.
  // - Loops over each pixel row-by-row.
  // - For each pixel:
  // - - Shoots samples_per_pixel rays with slight jitter.
  // - - Averages their returned colors.
  // - - Writes the result using write_color(...).
  void render(HittableList &world, HittableList &lights) override;

private:
  void render_cpu(HittableList &world, HittableList &lights);
  void render_gpu(HittableList &world, HittableList &lights);

private:
  // Camera config metadata.

  // Output image file name.
  std::string m_output_file;
};