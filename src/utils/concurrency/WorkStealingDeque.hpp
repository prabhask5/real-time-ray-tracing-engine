#pragma once

#include <atomic>

// Lock-free work-stealing deque based on Chase-Lev algorithm.
template <typename T, size_t SIZE = 1024> class WorkStealingDeque {
  static_assert((SIZE & (SIZE - 1)) == 0, "SIZE must be power of 2");

private:
  // This alignas(64) keyword forces this variable to be aligned on a 64-byte
  // boundary (a typical cache line size).

  // Represents the consumer pointer.
  alignas(64) std::atomic<int64_t> m_top{0};

  // Represents the producer pointer.
  alignas(64) std::atomic<int64_t> m_bottom{0};

  // Ring buffer to maintain lock-free.
  alignas(64) std::array<T, SIZE> m_buffer;

  // Fast modulo: index & m_mask  ==  index % SIZE.
  static constexpr int64_t m_mask = SIZE - 1;

public:
  WorkStealingDeque() = default;

  // Push to bottom (owner thread only).
  bool push(T &&item) {
    int64_t b = m_bottom.load(std::memory_order_relaxed);
    int64_t t = m_top.load(std::memory_order_acquire);

    if (b - t >= static_cast<int64_t>(SIZE)) {
      return false; // Full.
    }

    m_buffer[b & m_mask] = std::move(item);
    std::atomic_thread_fence(
        std::memory_order_release); // Enforces ordering of memory operations
                                    // across threads.
    m_bottom.store(b + 1, std::memory_order_relaxed);
    return true;
  }

  // Pop from bottom (owner thread only).
  bool pop(T &item) {
    int64_t b = m_bottom.load(std::memory_order_relaxed) - 1;
    m_bottom.store(b, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);

    int64_t t = m_top.load(std::memory_order_relaxed);

    if (t <= b) {
      if (t == b) {
        // Single element - compete with thieves.
        if (!m_top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                           std::memory_order_relaxed)) {
          m_bottom.store(b + 1, std::memory_order_relaxed);
          return false;
        }
        m_bottom.store(b + 1, std::memory_order_relaxed);
      }
      item = std::move(m_buffer[b & m_mask]);
      return true;
    } else {
      m_bottom.store(b + 1, std::memory_order_relaxed);
      return false;
    }
  }

  // Steal from top (any thread).
  bool steal(T &item) {
    int64_t t = m_top.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int64_t b = m_bottom.load(std::memory_order_acquire);

    if (t < b) {
      if (m_top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                        std::memory_order_relaxed)) {
        item = std::move(m_buffer[t & m_mask]);
        return true;
      }
    }
    return false;
  }

  bool empty() const {
    int64_t b = m_bottom.load(std::memory_order_relaxed);
    int64_t t = m_top.load(std::memory_order_relaxed);
    return b <= t;
  }
};