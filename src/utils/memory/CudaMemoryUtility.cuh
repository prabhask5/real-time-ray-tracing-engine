#pragma once

#ifdef USE_CUDA

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA error checking utility function.
inline void cuda_check_error(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),
            file, line);
    exit(EXIT_FAILURE);
  }
}

// Internal Macro for simplified error checking.
#define CUDA_CHECK(call) cuda_check_error((call), __FILE__, __LINE__)

// Safe CUDA memory allocation with error checking.
template <typename T> inline T *cudaMallocSafe(size_t count) {
  T *ptr = nullptr;
  size_t size = count * sizeof(T);
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
}

// Safe CUDA memory allocation with initialization to zero.
template <typename T> inline T *cudaMallocSafeZero(size_t count) {
  T *ptr = cudaMallocSafe<T>(count);
  size_t size = count * sizeof(T);
  CUDA_CHECK(cudaMemset(ptr, 0, size));
  return ptr;
}

// Safe CUDA memory copy from host to device.
template <typename T>
inline void cudaMemcpyHostToDeviceSafe(T *device_ptr, const T *host_ptr,
                                       size_t count) {
  size_t size = count * sizeof(T);
  CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice));
}

// Safe CUDA memory copy from device to host.
template <typename T>
inline void cudaMemcpyDeviceToHostSafe(T *host_ptr, const T *device_ptr,
                                       size_t count) {
  size_t size = count * sizeof(T);
  CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost));
}

// Safe CUDA memory copy from device to device.
template <typename T>
inline void cudaMemcpyDeviceToDeviceSafe(T *dest_ptr, const T *src_ptr,
                                         size_t count) {
  size_t size = count * sizeof(T);
  CUDA_CHECK(cudaMemcpy(dest_ptr, src_ptr, size, cudaMemcpyDeviceToDevice));
}

// Safe CUDA memory free.
inline void cudaFreeSafe(void *ptr) {
  if (ptr != nullptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
}

// CUDA device synchronization with error checking.
inline void cudaDeviceSynchronizeSafe() { CUDA_CHECK(cudaDeviceSynchronize()); }

// Get CUDA device properties.
inline cudaDeviceProp cudaGetDevicePropertiesSafe(int device = 0) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return prop;
}

// Set CUDA device.
inline void cudaSetDeviceSafe(int device) { CUDA_CHECK(cudaSetDevice(device)); }

// Get available memory on device.
inline void cudaMemGetInfoSafe(size_t *free_bytes, size_t *total_bytes) {
  CUDA_CHECK(cudaMemGetInfo(free_bytes, total_bytes));
}

#endif // USE_CUDA