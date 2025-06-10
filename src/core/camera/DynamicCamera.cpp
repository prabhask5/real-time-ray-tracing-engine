#include "DynamicCamera.hpp"
#include "../Ray.hpp"
#include <ColorUtility.hpp>
#include <Vec3Utility.hpp>

DynamicCamera::DynamicCamera(const CameraConfig &config)
    : Camera(config), m_window(nullptr), m_renderer(nullptr),
      m_texture(nullptr), m_font(nullptr), m_frame(0), m_last_fps_time(0),
      m_fps(0.0) {}

DynamicCamera::~DynamicCamera() {
  if (m_texture)
    SDL_DestroyTexture(m_texture);
  if (m_renderer)
    SDL_DestroyRenderer(m_renderer);
  if (m_window)
    SDL_DestroyWindow(m_window);
  if (m_font)
    TTF_CloseFont(m_font);
  TTF_Quit();
  SDL_Quit();
}

void DynamicCamera::render(const Hittable &world, const Hittable &lights) {
  initialize(); // Set up the camera basis vectors, viewplane, etc.

  // Initialize SDL subsystems.
  SDL_Init(SDL_INIT_VIDEO);
  TTF_Init();

  // Create SDL window and rendering components.
  m_window =
      SDL_CreateWindow("Dynamic Camera", m_image_width, m_image_height, 0);
  m_renderer = SDL_CreateRenderer(m_window, nullptr);
  m_texture = SDL_CreateTexture(m_renderer, SDL_PIXELFORMAT_RGB24,
                                SDL_TEXTUREACCESS_STREAMING, m_image_width,
                                m_image_height);

  // Load font for displaying FPS text.

  const char *try_paths[] = {
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", // Linux.
      "/usr/share/fonts/truetype/freefont/FreeSans.ttf", // Linux fallback.
      "/Library/Fonts/Arial.ttf",                     // macOS user-installed.
      "/System/Library/Fonts/Supplemental/Arial.ttf", // macOS system font.
      "/System/Library/Fonts/SFNS.ttf",               // macOS system UI font.
  };
  for (const char *path : try_paths) {
    m_font = TTF_OpenFont(path, 14);
    if (m_font)
      break;
  }

  // Allocate memory for color accumulation and pixel buffer (RGB8 per pixel).
  m_accumulation.assign(m_image_width * m_image_height, Color(0, 0, 0));
  m_pixels.assign(m_image_width * m_image_height * 3, 0);
  m_frame = 0;
  m_last_fps_time = SDL_GetTicks(); // For measuring FPS.

  bool running = true;
  int tile = 0;
  int tile_size =
      32; // Number of pixels per tile (used for progressive rendering).

  while (running) {
    handle_events(
        running); // Poll user input (WASD for movement, ESC to quit, etc.)
    if (!running)
      break;

    // Compute tile bounds for partial image updates.
    int tiles_x = (m_image_width + tile_size - 1) / tile_size;
    int start_x = (tile % tiles_x) * tile_size;
    int start_y = (tile / tiles_x) * tile_size;
    int end_x = std::min(start_x + tile_size, m_image_width);
    int end_y = std::min(start_y + tile_size, m_image_height);

    // Ray trace the current tile, accumulate color per pixel.
    for (int j = start_y; j < end_y; ++j) {
      for (int i = start_x; i < end_x; ++i) {
        // Shoots m_samples_per_pixel rays per pixel in a stratified grid
        // pattern.
        int sqrt_spp = int(std::sqrt(m_samples_per_pixel));
        for (int s_j = 0; s_j < sqrt_spp; s_j++) {
          for (int s_i = 0; s_i < sqrt_spp; s_i++) {
            // Shoots a ray through a random subpixel region, stratified by s_i
            // and s_j. This simulates camera defocus (depth of field) and
            // performs anti-aliasing by jittering the ray within the pixel
            // grid.
            Ray ray = get_ray(i, j, s_i, s_j);

            Color sample = ray_color(ray, m_max_depth, world,
                                     lights); // Trace ray through the scene.
            m_accumulation[j * m_image_width + i] +=
                sample; // Accumulate sample.
          }
        }
      }
    }

    // Move to next tile. If finished a full frame, increment frame counter.
    tile++;
    if (tile >= tiles_x * ((m_image_height + tile_size - 1) / tile_size)) {
      tile = 0;
      m_frame++;
    }

    // Convert accumulated color buffer to byte pixels and upload to texture.
    update_texture();

    // Update FPS counter every second.
    Uint64 now = SDL_GetTicks();
    if (now - m_last_fps_time >= 1000) {
      m_fps = 1000.0 * m_frame / (now - m_last_fps_time); // Frames per second.
      m_frame = 0;
      m_last_fps_time = now;
    }

    // Render to screen: clear, draw current image, draw FPS overlay.
    SDL_RenderClear(m_renderer);
    SDL_RenderTexture(m_renderer, m_texture, nullptr, nullptr);
    draw_fps();                    // Renders the FPS text in top right corner.
    SDL_RenderPresent(m_renderer); // Swap back buffer to screen.
  }
}

void DynamicCamera::handle_events(bool &running) {
  // Poll SDL events and handle user input for quitting, camera movement, and
  // render quality control.

  SDL_Event e;

  // Get the current keyboard state (returns a pointer to an array indexed by
  // SDL_Scancode).
  const bool *state = SDL_GetKeyboardState(nullptr);
  bool moved =
      false; // Tracks whether the camera moved, used to reset accumulation.

  // Process all queued SDL events.
  while (SDL_PollEvent(&e)) {
    // Handle quit event (e.g., window close).
    if (e.type == SDL_EVENT_QUIT)
      running = false;

    // Handle key press events (not held keys).
    if (e.type == SDL_EVENT_KEY_DOWN) {
      // ESC key: exit the application.
      if (e.key.key == SDLK_ESCAPE)
        running = false;

      // + key: increase samples per pixel (higher quality rendering).
      if (e.key.key == SDLK_EQUALS)
        m_samples_per_pixel++;

      // - key: decrease samples per pixel (but not below 1).
      if (e.key.key == SDLK_MINUS && m_samples_per_pixel > 1)
        m_samples_per_pixel--;
    }
  }

  // Movement speed per frame.
  double step = 10.0;

  // Move camera forward (W key).
  if (state[SDL_SCANCODE_W]) {
    m_lookfrom += Vec3(0, 0, step);
    m_lookat += Vec3(0, 0, step);
    moved = true;
  }

  // Move camera backward (S key).
  if (state[SDL_SCANCODE_S]) {
    m_lookfrom += Vec3(0, 0, -step);
    m_lookat += Vec3(0, 0, -step);
    moved = true;
  }

  // Move camera left (A key).
  if (state[SDL_SCANCODE_A]) {
    m_lookfrom += Vec3(-step, 0, 0);
    m_lookat += Vec3(-step, 0, 0);
    moved = true;
  }

  // Move camera right (D key).
  if (state[SDL_SCANCODE_D]) {
    m_lookfrom += Vec3(step, 0, 0);
    m_lookat += Vec3(step, 0, 0);
    moved = true;
  }

  // If the camera moved, reset accumulation buffer and frame counter to
  // re-render scene.
  if (moved) {
    std::fill(m_accumulation.begin(), m_accumulation.end(),
              Color(0, 0, 0)); // Clear accumulated colors.
    m_frame = 0;               // Reset frame count.
    initialize(); // Recalculate camera basis vectors and parameters.
  }
}

void DynamicCamera::update_texture() {
  // Calculate the total number of pixels in the rendered image.
  int total = m_image_width * m_image_height;

  // Compute a scale factor to average accumulated color samples over multiple
  // frames and across all samples per pixel. Prevent division by zero by
  // clamping the frame count to at least 1.
  double scale = m_pixel_samples_scale / std::max(1, m_frame);

  // Loop through every pixel in the image.
  for (int i = 0; i < total; ++i) {
    // Scale down the accumulated color at this pixel to compute the average.
    Color col = m_accumulation[i] * scale;

    // Convert each color channel (RGB) from floating point [0.0, 1.0] to 8-bit
    // integer [0, 255], and write the result into the linear pixel buffer
    // (m_pixels).
    m_pixels[i * 3 + 0] = to_byte(col.x()); // Red channel.
    m_pixels[i * 3 + 1] = to_byte(col.y()); // Green channel.
    m_pixels[i * 3 + 2] = to_byte(col.z()); // Blue channel.
  }

  // Upload the pixel data to the GPU texture.
  // The pitch is the number of bytes per row: width * 3 (for RGB).
  // The nullptr for the rect argument means the entire texture is updated.
  SDL_UpdateTexture(m_texture, nullptr, m_pixels.data(), m_image_width * 3);
}

void DynamicCamera::draw_fps() {
  // Early return if the font wasn't successfully loaded.
  if (!m_font)
    return;

  // Format the FPS value into a string buffer.
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%0.1f fps", m_fps);

  // Define a white color to render the text in.
  SDL_Color white{256, 256, 256, 256};

  // Create a surface containing the rendered text using anti-aliased blended
  // mode. This creates a temporary software surface.
  SDL_Surface *surface =
      TTF_RenderText_Blended(m_font, buffer, strlen(buffer), white);
  if (!surface)
    return;

  // Convert the surface to a hardware-accelerated texture.
  SDL_Texture *text = SDL_CreateTextureFromSurface(m_renderer, surface);

  // Set up the destination rectangle where the text will be drawn.
  SDL_FRect dst;
  dst.w = surface->w; // Width of the text.
  dst.h = surface->h; // Height of the text.
  dst.x = m_image_width - dst.w -
          10; // Align to top-right corner with 10px padding.
  dst.y = 10; // 10px from the top.

  SDL_DestroySurface(surface); // Free the software surface (not needed after
                               // texture is created).

  // Copy the texture to the renderer at the specified destination rectangle.
  SDL_RenderTexture(m_renderer, text, nullptr, &dst);

  SDL_DestroyTexture(text); // Free the texture to avoid memory leaks.
}
