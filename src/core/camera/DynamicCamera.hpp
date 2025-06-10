#pragma once

#include "Camera.hpp"
#include "CameraConfig.hpp"
#include <SDL3/SDL.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <vector>

class DynamicCamera : public Camera {
public:
  DynamicCamera(const CameraConfig &config);

  ~DynamicCamera();

  // The main interactive rendering loop:
  // - Initializes SDL, TTF, and internal camera state.
  // - Creates an SDL window, renderer, and streaming texture.
  // - Repeatedly renders the scene in small tiles, accumulating color per
  // pixel.
  // - Supports interactive movement (WASD) and sample count adjustment (+/-).
  // - Tracks and displays frames-per-second (FPS) using an on-screen overlay.
  // - Continues until the user closes the window or presses ESC.
  void render(HittableList &world, HittableList &lights) override;

private:
  // Constants.

  static const int MIN_TILE_SIZE = 16;
  static const int DEFAULT_TILE_SIZE = 32;
  static const int MAX_TILE_SIZE = 64;

  // Internal metadata for the dynamic camera.

  // Temporarily accumulates the running average of pixel colors over multiple
  // frames for progressive rendering. As the camera stops moving, you may want
  // to refine the image over time by averaging more samples per pixel.
  // When the camera moves, this buffer is reset to start fresh.
  std::vector<Color> m_accumulation;

  // Stores the final image data (RGB or RGBA) in 8-bit format to upload into
  // the SDL_Texture. You copy the averaged color from m_accumulation here after
  // converting it to [0–255] range.
  std::vector<unsigned char> m_pixels;

  // Counts how many samples of the pixel have already been taken for
  // progressive rendering. We stop sampling once we've converged onto the num
  // samples per pixel value. Used to calculate the running average of color in
  // m_accumulation. Reset to 0 if camera moves or scene changes.
  int m_samples_taken;

  // Counts how many frames have been rendered, used to calculate FPS.
  int m_frame;

  // Stores the last time (in ticks) when the FPS counter was updated.
  // Used with SDL_GetTicks64() to measure elapsed milliseconds between frames
  // or updates. Useful for throttling the FPS update (e.g., update once every
  // second).
  Uint64 m_last_fps_time;

  // Stores the current frames per second as a float.
  // Rendered on screen.
  double m_fps;

  // Tile size to progressively render. We will dynamically change this value
  // depending on FPS value currently. If FPS is high -> double to 64. If FPS is
  // low -> halve to 16.
  int m_tile_size;

  // SDL Objects.

  // Represents the main application window.
  SDL_Window *m_window;

  // Used to draw graphics (textures, shapes, images) onto the SDL_Window.
  SDL_Renderer *m_renderer;

  // Represents a 2D image or pixel buffer that can be rendered by SDL_Renderer.
  // For ray tracing, this is the rendered pixel buffer (uint32_t*, for
  // instance).
  SDL_Texture *m_texture;

  // Represents a loaded font for rendering text with SDL_ttf.
  TTF_Font *m_font;

private:
  // Handles any event (EXIT, etc.) or user input to the SDL window. Whenever we
  // move around the scene, we restart ray tracing to progessively view the
  // scene from a different location. This is also an optimization step to make
  // sure we don't compute too much whenever we're just moving around.
  void handle_events(bool &running);

  // Updates the pixels array to draw to the screen based on current Color
  // accumulation per pixel, calculated from progressive ray tracing over time.
  void update_texture();

  // Directly draws the value in m_fps to the SDL window.
  void draw_fps(bool converged);
};