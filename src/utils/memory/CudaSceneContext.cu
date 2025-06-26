#ifdef USE_CUDA

#include "CudaSceneContext.cuh"

// Static member definition.
CudaSceneContext *CudaSceneContext::context = nullptr;

// Device context pointer definition.
__device__ CudaSceneContextView *d_scene_context = nullptr;

// Implementation of global device functions.
__device__ const CudaMaterial &cuda_get_material(size_t index) {
  return d_scene_context->materials[index];
}

__device__ const CudaTexture &cuda_get_texture(size_t index) {
  return d_scene_context->textures[index];
}

void initialize_device_scene_context() {
  CudaSceneContext &host_context = CudaSceneContext::get_context();

  // Get the raw device pointer to the uploaded view struct.
  CudaSceneContextView *context_view_ptr = host_context.get_device_view();

  // Set the device symbol.
  CUDA_CHECK(cudaMemcpyToSymbol(d_scene_context, &context_view_ptr,
                                sizeof(CudaSceneContextView *)));
}

void cleanup_device_scene_context() {
  // Reset device symbol to nullptr.
  CudaSceneContextView *null_ptr = nullptr;
  CUDA_CHECK(cudaMemcpyToSymbol(d_scene_context, &null_ptr,
                                sizeof(CudaSceneContextView *)));

  // Note: The actual device context memory is managed by CudaUniquePtr
  // in CudaSceneContext and will be automatically freed.
}

#endif // USE_CUDA