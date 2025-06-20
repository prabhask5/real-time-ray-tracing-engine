#pragma once

#include "WorkerThread.hpp"

// Lock-free work-stealing thread pool implementation.
class ThreadPool {
private:
  std::vector<std::unique_ptr<WorkerThread>> m_workers;
  std::atomic<bool> m_stop{true};
  std::atomic<size_t> m_active_workers{0};
  std::atomic<size_t> m_pending_tasks{0};
  size_t m_thread_count;

  // Global overflow queue for when local queues are full.
  WorkStealingDeque<Task> m_global_queue;

  // Helper function to get thread-local worker ID.
  static size_t &get_worker_id() {
    static thread_local size_t worker_id = SIZE_MAX;
    return worker_id;
  }

  // CPU pause instruction for spin-wait optimization.
  void cpu_pause() {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
  }

  void worker_loop(size_t worker_id) {
    get_worker_id() = worker_id;
    auto &worker = *m_workers[worker_id];

    worker.active.store(true, std::memory_order_relaxed);
    m_active_workers.fetch_add(1, std::memory_order_relaxed);

    while (!m_stop.load(std::memory_order_relaxed)) {
      Task task;
      bool found_work = false;

      // 1. Try local queue first
      if (worker.local_queue.pop(task)) {
        found_work = true;
      }
      // 2. Try global queue
      else if (m_global_queue.steal(task)) {
        found_work = true;
      }
      // 3. Try to steal from other workers
      else {
        for (size_t i = 1; i <= m_thread_count && !found_work; ++i) {
          size_t steal_from = (worker_id + i) % m_thread_count;
          if (m_workers[steal_from]->local_queue.steal(task)) {
            found_work = true;
            break;
          }
        }
      }

      if (found_work) {
        m_pending_tasks.fetch_sub(1, std::memory_order_relaxed);
        try {
          task();
        } catch (...) {
          // Swallow exceptions to prevent thread termination.
        }
      } else {
        // No work found - progressive backoff.
        for (int i = 0; i < 16 && !m_stop.load(std::memory_order_relaxed);
             ++i) {
          cpu_pause();
        }
        if (!m_stop.load(std::memory_order_relaxed)) {
          std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
      }
    }

    worker.active.store(false, std::memory_order_relaxed);
    m_active_workers.fetch_sub(1, std::memory_order_relaxed);
  }

public:
  explicit ThreadPool(size_t thread_count = std::thread::hardware_concurrency())
      : m_thread_count(thread_count) {
    m_workers.reserve(m_thread_count);
    for (size_t i = 0; i < m_thread_count; ++i) {
      m_workers.emplace_back(std::make_unique<WorkerThread>());
    }
  }

  ~ThreadPool() { finish(); }

  // Non-copyable, non-movable.
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;
  ThreadPool(ThreadPool &&) = delete;
  ThreadPool &operator=(ThreadPool &&) = delete;

  void start() {
    if (!m_stop.exchange(false, std::memory_order_relaxed)) {
      return; // Already started.
    }

    for (size_t i = 0; i < m_thread_count; ++i) {
      m_workers[i]->thread = std::thread(&ThreadPool::worker_loop, this, i);
    }
  }

  void submit_job(Task task) {
    if (m_stop.load(std::memory_order_relaxed)) {
      return;
    }

    m_pending_tasks.fetch_add(1, std::memory_order_relaxed);

    // Try to push to current worker's local queue first.
    size_t current_worker_id = get_worker_id();
    if (current_worker_id < m_thread_count) {
      if (m_workers[current_worker_id]->local_queue.push(std::move(task))) {
        return;
      }
    }

    // Fallback to global queue.
    if (!m_global_queue.push(std::move(task))) {
      // Global queue full - execute immediately on current thread.
      m_pending_tasks.fetch_sub(1, std::memory_order_relaxed);
      try {
        task();
      } catch (...) {
        // Swallow exceptions.
      }
    }
  }

  void finish() {
    if (m_stop.load(std::memory_order_relaxed)) {
      return;
    }

    // Wait for all tasks to complete.
    while (m_pending_tasks.load(std::memory_order_relaxed) > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    // Signal shutdown.
    m_stop.store(true, std::memory_order_relaxed);

    // Join all threads.
    for (auto &worker : m_workers) {
      if (worker->thread.joinable()) {
        worker->thread.join();
      }
    }
  }

  // Utility methods.
  size_t get_thread_count() const { return m_thread_count; }
  size_t get_active_workers() const {
    return m_active_workers.load(std::memory_order_relaxed);
  }
  size_t get_pending_tasks() const {
    return m_pending_tasks.load(std::memory_order_relaxed);
  }
  bool is_running() const { return !m_stop.load(std::memory_order_relaxed); }
};
