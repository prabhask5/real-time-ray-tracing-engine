#pragma once

#ifdef USE_CUDA

#include "../../utils/math/PDF.cuh"
#include "CudaMemoryUtility.cuh"
#include "CudaUniquePtr.cuh"
#include <memory>
#include <vector>

// Forward declarations.
struct CudaMaterial;
struct CudaTexture;
struct CudaPDF;

// Simple device context struct for kernel access.
struct CudaSceneContextView {
  const CudaMaterial *materials;
  const CudaTexture *textures;
  size_t material_count;
  size_t texture_count;
};

// Singleton class to manage CUDA scene resources.
class CudaSceneContext {
private:
  static CudaSceneContext *context;

  // Host storage for scene data.
  std::vector<CudaMaterial> m_host_materials;
  std::vector<CudaTexture> m_host_textures;

  // Device storage with smart pointers.
  // NOTE: This is still needed with the View struct for device/kernel access
  // because these unique pointers manage the lifetime and allocation of the GPU
  // memory, the View struct is just a View into that memory using a pointer
  // that is accessible on the device side, the host still manages the memory.
  CudaUniquePtr<CudaMaterial> m_device_materials;
  CudaUniquePtr<CudaTexture> m_device_textures;
  CudaUniquePtr<CudaSceneContextView> m_device_view;

  // Counts.
  size_t m_material_count;
  size_t m_texture_count;

  // Single large buffer for sub-allocation.
  CudaUniquePtr<char> m_device_buffer;
  size_t m_buffer_size;
  size_t m_buffer_offset;

  CudaSceneContext()
      : m_material_count(0), m_texture_count(0), m_buffer_size(0),
        m_buffer_offset(0) {}

public:
  // Singleton access.
  static CudaSceneContext &get_context() {
    if (context == nullptr) {
      context = new CudaSceneContext();
    }
    return *context;
  }

  // Destroy singleton (call before program exit).
  static void destroy_context() {
    delete context;
    context = nullptr;
  }

  // Initialize the scene context with estimated counts.
  void initialize(size_t estimated_materials, size_t estimated_textures,
                  size_t buffer_size_mb = 128) {
    // Reserve host storage.
    m_host_materials.reserve(estimated_materials);
    m_host_textures.reserve(estimated_textures);

    // Calculate buffer size.
    m_buffer_size = buffer_size_mb * 1024 * 1024; // Convert MB to bytes.
    m_buffer_offset = 0;

    // Allocate large buffer for sub-allocation.
    m_device_buffer = make_cuda_unique_array_zero<char>(m_buffer_size);
  }

  // Add material and return index.
  size_t add_material(const CudaMaterial &material) {
    size_t index = m_host_materials.size();
    m_host_materials.push_back(material);
    return index;
  }

  // Add texture and return index.
  size_t add_texture(const CudaTexture &texture) {
    size_t index = m_host_textures.size();
    m_host_textures.push_back(texture);
    return index;
  }

  // Finalize and upload all data to GPU.
  void finalize_and_upload() {
    m_material_count = m_host_materials.size();
    m_texture_count = m_host_textures.size();

    // Upload materials.
    if (m_material_count > 0) {
      m_device_materials =
          make_cuda_unique_from_host(m_host_materials.data(), m_material_count);
    }

    // Upload textures.
    if (m_texture_count > 0) {
      m_device_textures =
          make_cuda_unique_from_host(m_host_textures.data(), m_texture_count);
    }

    // Create and upload device context view.
    CudaSceneContextView context_view = {.materials = m_device_materials.get(),
                                         .textures = m_device_textures.get(),
                                         .material_count = m_material_count,
                                         .texture_count = m_texture_count};

    m_device_view = make_cuda_unique_from_host(&context_view, 1);
  }

  // Get context view pointer (for kernel access).
  CudaSceneContextView *get_device_view() const { return m_device_view.get(); }

  // Get device pointers (for kernel access).
  __host__ __device__ const CudaMaterial *get_materials() const {
    return m_device_materials.get();
  }

  __host__ __device__ const CudaTexture *get_textures() const {
    return m_device_textures.get();
  }

  // Get counts.
  size_t get_material_count() const { return m_material_count; }
  size_t get_texture_count() const { return m_texture_count; }

  // Get host arrays (for JSON serialization).
  const std::vector<CudaMaterial> &get_host_materials() const {
    return m_host_materials;
  }
  const std::vector<CudaTexture> &get_host_textures() const {
    return m_host_textures;
  }

  // Sub-allocate from the large buffer.
  template <typename T> T *suballocate(size_t count) {
    size_t size = count * sizeof(T);

    // Align to 256 bytes for optimal GPU memory access.
    size_t aligned_size = (size + 255) & ~255;

    if (!m_device_buffer) {
      fprintf(stderr, "ERROR: CUDA device buffer not allocated\n");
      exit(EXIT_FAILURE);
    }

    if (m_buffer_offset + aligned_size > m_buffer_size) {
      fprintf(
          stderr,
          "CUDA buffer overflow: requested %zu bytes, available %zu bytes\n",
          aligned_size, m_buffer_size - m_buffer_offset);
      exit(EXIT_FAILURE);
    }

    T *ptr = reinterpret_cast<T *>(m_device_buffer.get() + m_buffer_offset);
    m_buffer_offset += aligned_size;
    return ptr;
  }

  // Get remaining buffer space.
  size_t get_free_buffer_space() const {
    return m_buffer_size - m_buffer_offset;
  }

  // Reset buffer offset for reuse.
  void reset_buffer_offset() { m_buffer_offset = 0; }
};

// Include complete type definitions for inline functions below
#include "../../scene/materials/Material.cuh"
#include "../../scene/textures/Texture.cuh"

// Global device functions to access materials, textures, and PDFs.
__device__ const CudaMaterial &cuda_get_material(size_t index);
__device__ const CudaTexture &cuda_get_texture(size_t index);

// Device/kernel context pointer (to be set by initialization).
extern __device__ CudaSceneContextView *d_scene_context;

// Function to initialize device context pointer.
void initialize_device_scene_context();

// Function to cleanup device context.
void cleanup_device_scene_context();

#endif // USE_CUDA