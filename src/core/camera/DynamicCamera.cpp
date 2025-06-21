#include "DynamicCamera.hpp"
#include "../../optimization/BVHNode.hpp"
#include "../../utils/ColorUtility.hpp"
#include "../../utils/concurrency/ThreadPool.hpp"
#include "../../utils/math/Vec3Utility.hpp"
#include "../HittableList.hpp"
#include "../Ray.hpp"

#ifdef USE_CUDA
#include "../../scene/SceneConversions.cuh"
#include "CameraKernelWrappers.cuh"
#include "CameraKernels.cuh"
#include <ctime>
#endif

const int DynamicCamera::MIN_TILE_SIZE;
const int DynamicCamera::DEFAULT_TILE_SIZE;
const int DynamicCamera::MAX_TILE_SIZE;

DynamicCamera::DynamicCamera(const CameraConfig &config)
    : Camera(config), m_samples_taken(0), m_frame(0), m_last_fps_time(0),
      m_fps(0.0), m_tile_size(DEFAULT_TILE_SIZE), m_window(nullptr),
      m_renderer(nullptr), m_texture(nullptr), m_font(nullptr) {}

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

void DynamicCamera::render(HittableList &world, HittableList &lights) {
  if (m_use_gpu)
    render_gpu(world, lights);
  else
    render_cpu(world, lights);
}

void DynamicCamera::render_cpu(HittableList &world, HittableList &lights) {
  initialize(); // Set up the camera basis vectors, viewplane, etc.

  if (m_use_bvh) {
    if (!world.get_objects().empty())
      world = HittableList(std::make_shared<BVHNode>(world));
    if (!lights.get_objects().empty())
      lights = HittableList(std::make_shared<BVHNode>(lights));
  }

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
  m_last_fps_time = SDL_GetTicks(); // For measuring FPS.

  ThreadPool pool(std::thread::hardware_concurrency());

  bool running = true;

  while (running) {
    handle_events(
        running); // Poll user input (WASD for movement, ESC to quit, etc.)
    if (!running)
      break;

    // Recompute sampling and tiling parameters in case the user changed
    // samples-per-pixel or the tile size was adjusted dynamically.
    int sqrt_spp = int(std::sqrt(m_samples_per_pixel));
    int total_strata = sqrt_spp * sqrt_spp;
    int num_tiles_x = (m_image_width + m_tile_size - 1) / m_tile_size;
    int num_tiles_y = (m_image_height + m_tile_size - 1) / m_tile_size;

    // Stop sampling after convergence.
    bool converged = (m_samples_taken >= total_strata);

    if (!converged) {
      for (int tile = 0; tile < (num_tiles_x * num_tiles_y); ++tile) {
        // Compute tile bounds for partial image updates.
        int tile_r = tile / num_tiles_x;
        int tile_c = tile % num_tiles_x;
        int start_r = tile_r * m_tile_size;
        int start_c = tile_c * m_tile_size;
        int end_r = std::min(start_r + m_tile_size, m_image_height);
        int end_c = std::min(start_c + m_tile_size, m_image_width);

        // Compute stratified sample index for this frame.
        int s_i = m_samples_taken % sqrt_spp;
        int s_j = m_samples_taken / sqrt_spp;

        if (m_use_parallelism) {
          pool.start();

          // Ray trace the current tile, accumulate color per pixel.
          for (int j = start_r; j < end_r; ++j) {
            pool.submit_job([this, &world, &lights, tile_r, tile_c, start_c,
                             end_c, j, s_i, s_j]() {
              for (int i = start_c; i < end_c; ++i) {
                // Shoots a ray through a random subpixel region, stratified
                // by s_i
                // and s_j. This simulates camera defocus (depth of field) and
                // performs anti-aliasing by jittering the ray within the
                // pixel grid.
                Ray ray = get_ray(i, j, s_i, s_j);

                // Trace ray through the scene.
                Color sample = ray_color(ray, m_max_depth, world, lights);

                // Accumulate sample.
                m_accumulation[j * m_image_width + i] += sample;
              }
            });
          }

          pool.finish();
        } else {
          // Ray trace the current tile, accumulate color per pixel.
          for (int j = start_r; j < end_r; ++j) {
            for (int i = start_c; i < end_c; ++i) {
              // Shoots a ray through a random subpixel region, stratified by
              // s_i and s_j. This simulates camera defocus (depth of field) and
              // performs anti-aliasing by jittering the ray within the pixel
              // grid.
              Ray ray = get_ray(i, j, s_i, s_j);

              // Trace ray through the scene.
              Color sample = ray_color(ray, m_max_depth, world, lights);

              // Accumulate sample.
              m_accumulation[j * m_image_width + i] += sample;
            }
          }
        }
      }
    }

    // Increment frame and sample counter.
    if (!converged)
      m_samples_taken++;
    m_frame++;

    // Convert accumulated color buffer to byte pixels and upload to texture.
    update_texture();

    // Update FPS counter every second.
    Uint64 now = SDL_GetTicks();
    Uint64 elapsed = now - m_last_fps_time;
    if (elapsed >= 1000) {
      m_fps = 1000.0 * m_frame / elapsed; // Frames per second.
      m_frame = 0;
      m_last_fps_time = now;

      // Adjust tile size dynamically based on FPS.
      if (!converged && m_fps > 30.0 && m_tile_size < MAX_TILE_SIZE)
        m_tile_size = std::min(m_tile_size * 2, MAX_TILE_SIZE);
      else if (!converged && m_fps < 15.0 && m_tile_size > MIN_TILE_SIZE)
        m_tile_size = std::max(m_tile_size / 2, MIN_TILE_SIZE);
    }

    // Render to screen: clear, draw current image, draw FPS overlay.
    SDL_RenderClear(m_renderer);
    SDL_RenderTexture(m_renderer, m_texture, nullptr, nullptr);
    draw_fps(converged);           // Renders the FPS text in top right corner.
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
    m_samples_taken =
        0;        // Reset number of samples taken to restart ray tracing.
    initialize(); // Recalculate camera basis vectors and parameters.
  }
}

void DynamicCamera::update_texture() {
  // Calculate the total number of pixels in the rendered image.
  int total = m_image_width * m_image_height;

  // Compute a scale factor to average accumulated color samples over multiple
  // frames. Prevent division by zero by
  // clamping the samples taken count to at least 1.
  double scale = 1.0 / std::max(1, m_samples_taken);

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

void DynamicCamera::draw_fps(bool converged) {
  // Early return if the font wasn't successfully loaded.
  if (!m_font)
    return;

  // Compose the label: FPS + (optional) convergence marker.
  char buffer[64];
  if (converged)
    std::snprintf(buffer, sizeof(buffer), "%0.1f fps  ✓ Converged", m_fps);
  else
    std::snprintf(buffer, sizeof(buffer), "%0.1f fps", m_fps);

  // Define a white color to render the text in.
  SDL_Color white{255, 255, 255, 255};

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

void DynamicCamera::render_gpu(HittableList &world, HittableList &lights) {
#ifdef USE_CUDA
  initialize();

  if (m_use_bvh) {
    if (!world.get_objects().empty())
      world = HittableList(std::make_shared<BVHNode>(world));
    if (!lights.get_objects().empty())
      lights = HittableList(std::make_shared<BVHNode>(lights));
  }

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
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
      "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
      "/Library/Fonts/Arial.ttf",
      "/System/Library/Fonts/Supplemental/Arial.ttf",
      "/System/Library/Fonts/SFNS.ttf",
  };
  for (const char *path : try_paths) {
    m_font = TTF_OpenFont(path, 14);
    if (m_font)
      break;
  }

  // Allocate memory for color accumulation and pixel buffer.
  m_accumulation.assign(m_image_width * m_image_height, Color(0, 0, 0));
  m_pixels.assign(m_image_width * m_image_height * 3, 0);
  m_last_fps_time = SDL_GetTicks();

  // CUDA setup.
  CudaColor *d_accumulation;
  curandState *d_rand_states;
  size_t accumulation_size = m_image_width * m_image_height * sizeof(CudaColor);
  size_t rand_states_size =
      m_image_width * m_image_height * sizeof(curandState);

  if (cudaMalloc(&d_accumulation, accumulation_size) != cudaSuccess) {
    std::cerr << "Failed to allocate CUDA memory for accumulation buffer"
              << std::endl;
    render_cpu(world, lights);
    return;
  }
  if (cudaMalloc(&d_rand_states, rand_states_size) != cudaSuccess) {
    std::cerr << "Failed to allocate CUDA memory for random states"
              << std::endl;
    cudaFree(d_accumulation);
    render_cpu(world, lights);
    return;
  }

  // Initialize CUDA accumulation buffer.
  cudaMemset(d_accumulation, 0, accumulation_size);

  // Initialize random states.
  dim3 block_size(16, 16);
  dim3 grid_size((m_image_width + block_size.x - 1) / block_size.x,
                 (m_image_height + block_size.y - 1) / block_size.y);

  cuda_init_rand_states_wrapper(d_rand_states, m_image_width, m_image_height,
                                (unsigned long)time(nullptr), grid_size,
                                block_size);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "Failed to initialize CUDA random states" << std::endl;
    cudaFree(d_accumulation);
    cudaFree(d_rand_states);
    render_cpu(world, lights);
    return;
  }

  // Convert CPU objects to CUDA format using comprehensive conversion.
  CudaSceneData cuda_scene_data = convert_complete_scene_to_cuda(world, lights);
  if (cuda_scene_data.world_objects_buffer == nullptr ||
      cuda_scene_data.lights_objects_buffer == nullptr) {
    std::cerr << "Failed to convert scene to CUDA format" << std::endl;
    cudaFree(d_accumulation);
    cudaFree(d_rand_states);
    render_cpu(world, lights);
    return;
  }
  CudaHittable cuda_world = cuda_scene_data.world;
  CudaHittable cuda_lights = cuda_scene_data.lights;

  bool running = true;

  while (running) {
    handle_events(running);
    if (!running)
      break;

    int sqrt_spp = int(std::sqrt(m_samples_per_pixel));
    int total_strata = sqrt_spp * sqrt_spp;
    int num_tiles_x = (m_image_width + m_tile_size - 1) / m_tile_size;
    int num_tiles_y = (m_image_height + m_tile_size - 1) / m_tile_size;

    bool converged = (m_samples_taken >= total_strata);

    if (!converged) {
      for (int tile = 0; tile < (num_tiles_x * num_tiles_y); ++tile) {
        int tile_r = tile / num_tiles_x;
        int tile_c = tile % num_tiles_x;
        int start_r = tile_r * m_tile_size;
        int start_c = tile_c * m_tile_size;
        int end_r = std::min(start_r + m_tile_size, m_image_height);
        int end_c = std::min(start_c + m_tile_size, m_image_width);

        int s_i = m_samples_taken % sqrt_spp;
        int s_j = m_samples_taken / sqrt_spp;

        // Launch CUDA kernel for this tile.
        dim3 tile_block_size(16, 16);
        dim3 tile_grid_size(
            (end_c - start_c + tile_block_size.x - 1) / tile_block_size.x,
            (end_r - start_r + tile_block_size.y - 1) / tile_block_size.y);

        // Convert camera parameters to CUDA format.
        CudaPoint3 cuda_center(m_center.x(), m_center.y(), m_center.z());
        CudaPoint3 cuda_pixel00_loc(m_pixel00_loc.x(), m_pixel00_loc.y(),
                                    m_pixel00_loc.z());
        CudaVec3 cuda_pixel_delta_u(m_pixel_delta_u.x(), m_pixel_delta_u.y(),
                                    m_pixel_delta_u.z());
        CudaVec3 cuda_pixel_delta_v(m_pixel_delta_v.x(), m_pixel_delta_v.y(),
                                    m_pixel_delta_v.z());
        CudaVec3 cuda_u(m_u.x(), m_u.y(), m_u.z());
        CudaVec3 cuda_v(m_v.x(), m_v.y(), m_v.z());
        CudaVec3 cuda_w(m_w.x(), m_w.y(), m_w.z());
        CudaVec3 cuda_defocus_disk_u(m_defocus_disk_u.x(), m_defocus_disk_u.y(),
                                     m_defocus_disk_u.z());
        CudaVec3 cuda_defocus_disk_v(m_defocus_disk_v.x(), m_defocus_disk_v.y(),
                                     m_defocus_disk_v.z());
        CudaColor cuda_background(m_background.x(), m_background.y(),
                                  m_background.z());

        cuda_dynamic_render_tile_wrapper(
            d_accumulation, m_image_width, m_image_height, start_r, end_r,
            start_c, end_c, s_i, s_j, sqrt_spp, m_max_depth, cuda_center,
            cuda_pixel00_loc, cuda_pixel_delta_u, cuda_pixel_delta_v, cuda_u,
            cuda_v, cuda_w, cuda_defocus_disk_u, cuda_defocus_disk_v,
            m_defocus_angle, cuda_background, cuda_world, cuda_lights,
            d_rand_states, tile_grid_size, tile_block_size);

        cudaDeviceSynchronize();
      }
    }

    if (!converged)
      m_samples_taken++;
    m_frame++;

    // Copy accumulation buffer back to CPU.
    std::vector<CudaColor> cuda_accumulation(m_image_width * m_image_height);
    cudaMemcpy(cuda_accumulation.data(), d_accumulation, accumulation_size,
               cudaMemcpyDeviceToHost);

    // Convert CUDA accumulation to CPU format.
    for (int i = 0; i < m_image_width * m_image_height; ++i) {
      m_accumulation[i] = Color(cuda_accumulation[i].x, cuda_accumulation[i].y,
                                cuda_accumulation[i].z);
    }

    update_texture();

    // Update FPS counter.
    Uint64 now = SDL_GetTicks();
    Uint64 elapsed = now - m_last_fps_time;
    if (elapsed >= 1000) {
      m_fps = 1000.0 * m_frame / elapsed;
      m_frame = 0;
      m_last_fps_time = now;

      if (!converged && m_fps > 30.0 && m_tile_size < MAX_TILE_SIZE)
        m_tile_size = std::min(m_tile_size * 2, MAX_TILE_SIZE);
      else if (!converged && m_fps < 15.0 && m_tile_size > MIN_TILE_SIZE)
        m_tile_size = std::max(m_tile_size / 2, MIN_TILE_SIZE);
    }

    SDL_RenderClear(m_renderer);
    SDL_RenderTexture(m_renderer, m_texture, nullptr, nullptr);
    draw_fps(converged);
    SDL_RenderPresent(m_renderer);
  }

  // Cleanup CUDA memory and scene data.
  cleanup_cuda_scene_data(cuda_scene_data);
  cudaFree(d_accumulation);
  cudaFree(d_rand_states);
#else
  // Fall back to CPU rendering if CUDA is not available.
  render_cpu(world, lights);
#endif
}
