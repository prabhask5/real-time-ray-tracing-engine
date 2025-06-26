#pragma once

#ifdef USE_CUDA

#include "CudaMemoryUtility.cuh"
#include <memory>

// RAII CUDA smart pointer that automatically frees GPU memory.
template <typename T> class CudaUniquePtr {
private:
  T *m_ptr;

public:
  // Default constructor.
  CudaUniquePtr() : m_ptr(nullptr) {}

  // Constructor that takes ownership of raw pointer.
  explicit CudaUniquePtr(T *ptr) : m_ptr(ptr) {}

  // Destructor automatically frees memory.
  ~CudaUniquePtr() { reset(); }

  // Move constructor.
  CudaUniquePtr(CudaUniquePtr &&other) noexcept : m_ptr(other.m_ptr) {
    other.m_ptr = nullptr;
  }

  // Move assignment operator.
  CudaUniquePtr &operator=(CudaUniquePtr &&other) noexcept {
    if (this != &other) {
      reset();
      m_ptr = other.m_ptr;
      other.m_ptr = nullptr;
    }
    return *this;
  }

  // Deleted copy constructor and assignment to prevent copying.
  CudaUniquePtr(const CudaUniquePtr &) = delete;
  CudaUniquePtr &operator=(const CudaUniquePtr &) = delete;

  // Get raw pointer.
  T *get() const { return m_ptr; }

  // Release ownership and return raw pointer.
  T *release() {
    T *ptr = m_ptr;
    m_ptr = nullptr;
    return ptr;
  }

  // Reset with new pointer, freeing old memory.
  void reset(T *ptr = nullptr) {
    if (m_ptr != ptr) {
      cudaFreeSafe(m_ptr);
      m_ptr = ptr;
    }
  }

  // Boolean conversion for null checking.
  explicit operator bool() const { return m_ptr != nullptr; }

  // Dereference operators.
  T &operator*() const { return *m_ptr; }
  T *operator->() const { return m_ptr; }

  // Array access operator.
  T &operator[](size_t index) const { return m_ptr[index]; }
};

// Factory function to create CudaUniquePtr with single object.
template <typename T, typename... Args>
CudaUniquePtr<T> make_cuda_unique(Args &&...args) {
  T *ptr = cudaMallocSafe<T>(1);

  // Create object on host and copy to device.
  T host_obj(std::forward<Args>(args)...);
  cudaMemcpyHostToDeviceSafe(ptr, &host_obj, 1);

  return CudaUniquePtr<T>(ptr);
}

// Factory function to create CudaUniquePtr with array.
template <typename T> CudaUniquePtr<T> make_cuda_unique_array(size_t count) {
  T *ptr = cudaMallocSafe<T>(count);
  return CudaUniquePtr<T>(ptr);
}

// Factory function to create CudaUniquePtr with zero-initialized array.
template <typename T>
CudaUniquePtr<T> make_cuda_unique_array_zero(size_t count) {
  T *ptr = cudaMallocSafeZero<T>(count);
  return CudaUniquePtr<T>(ptr);
}

// Factory function to create CudaUniquePtr from host data.
template <typename T>
CudaUniquePtr<T> make_cuda_unique_from_host(const T *host_data, size_t count) {
  T *ptr = cudaMallocSafe<T>(count);
  cudaMemcpyHostToDeviceSafe(ptr, host_data, count);
  return CudaUniquePtr<T>(ptr);
}

#endif // USE_CUDA