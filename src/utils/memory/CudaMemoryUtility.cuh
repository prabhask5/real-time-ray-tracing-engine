#pragma once

#ifdef USE_CUDA

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

inline void cuda_check_error(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),
            file, line);
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK_AT(call, file, line) cuda_check_error((call), (file), (line))
#define CUDA_CHECK(call) CUDA_CHECK_AT((call), __FILE__, __LINE__)

template <typename T>
inline T *cudaMallocSafe_impl(size_t count, const char *file, int line) {
  T *ptr = nullptr;
  size_t size = count * sizeof(T);
  CUDA_CHECK_AT(cudaMalloc(&ptr, size), file, line);
  return ptr;
}
#define cudaMallocSafe(T, count)                                               \
  cudaMallocSafe_impl<T>((count), __FILE__, __LINE__)

template <typename T>
inline T *cudaMallocSafeZero_impl(size_t count, const char *file, int line) {
  T *ptr = cudaMallocSafe_impl<T>(count, file, line);
  size_t size = count * sizeof(T);
  CUDA_CHECK_AT(cudaMemset(ptr, 0, size), file, line);
  return ptr;
}
#define cudaMallocSafeZero(T, count)                                           \
  cudaMallocSafeZero_impl<T>((count), __FILE__, __LINE__)

template <typename T>
inline void cudaMemcpyHostToDeviceSafe_impl(T *device_ptr, const T *host_ptr,
                                            size_t count, const char *file,
                                            int line) {
  size_t size = count * sizeof(T);
  CUDA_CHECK_AT(cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice),
                file, line);
}
#define cudaMemcpyHostToDeviceSafe(device_ptr, host_ptr, count)                \
  cudaMemcpyHostToDeviceSafe_impl((device_ptr), (host_ptr), (count), __FILE__, \
                                  __LINE__)

template <typename T>
inline void cudaMemcpyDeviceToHostSafe_impl(T *host_ptr, const T *device_ptr,
                                            size_t count, const char *file,
                                            int line) {
  size_t size = count * sizeof(T);
  CUDA_CHECK_AT(cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost),
                file, line);
}
#define cudaMemcpyDeviceToHostSafe(host_ptr, device_ptr, count)                \
  cudaMemcpyDeviceToHostSafe_impl((host_ptr), (device_ptr), (count), __FILE__, \
                                  __LINE__)

template <typename T>
inline void cudaMemcpyDeviceToDeviceSafe_impl(T *dest_ptr, const T *src_ptr,
                                              size_t count, const char *file,
                                              int line) {
  size_t size = count * sizeof(T);
  CUDA_CHECK_AT(cudaMemcpy(dest_ptr, src_ptr, size, cudaMemcpyDeviceToDevice),
                file, line);
}
#define cudaMemcpyDeviceToDeviceSafe(dest_ptr, src_ptr, count)                 \
  cudaMemcpyDeviceToDeviceSafe_impl((dest_ptr), (src_ptr), (count), __FILE__,  \
                                    __LINE__)

inline void cudaFreeSafe_impl(void *ptr, const char *file, int line) {
  if (ptr != nullptr) {
    CUDA_CHECK_AT(cudaFree(ptr), file, line);
  }
}
#define cudaFreeSafe(ptr) cudaFreeSafe_impl((ptr), __FILE__, __LINE__)

inline void cudaDeviceSynchronizeSafe_impl(const char *file, int line) {
  CUDA_CHECK_AT(cudaDeviceSynchronize(), file, line);
}
#define cudaDeviceSynchronizeSafe()                                            \
  cudaDeviceSynchronizeSafe_impl(__FILE__, __LINE__)

inline cudaDeviceProp
cudaGetDevicePropertiesSafe_impl(int device, const char *file, int line) {
  cudaDeviceProp prop;
  CUDA_CHECK_AT(cudaGetDeviceProperties(&prop, device), file, line);
  return prop;
}
#define cudaGetDevicePropertiesSafe(device)                                    \
  cudaGetDevicePropertiesSafe_impl((device), __FILE__, __LINE__)

inline void cudaSetDeviceSafe_impl(int device, const char *file, int line) {
  CUDA_CHECK_AT(cudaSetDevice(device), file, line);
}
#define cudaSetDeviceSafe(device)                                              \
  cudaSetDeviceSafe_impl((device), __FILE__, __LINE__)

inline void cudaMemGetInfoSafe_impl(size_t *free_bytes, size_t *total_bytes,
                                    const char *file, int line) {
  CUDA_CHECK_AT(cudaMemGetInfo(free_bytes, total_bytes), file, line);
}
#define cudaMemGetInfoSafe(free_bytes, total_bytes)                            \
  cudaMemGetInfoSafe_impl((free_bytes), (total_bytes), __FILE__, __LINE__)

#endif // USE_CUDA