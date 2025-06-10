#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// Simple thread pool that runs a fixed number of worker threads.
// Jobs are executed as soon as a worker becomes available.
class ThreadPool {
public:
  ThreadPool(size_t thread_count) : m_thread_count(thread_count) {
    for (size_t i = 0; i < m_thread_count; ++i) {
      m_workers.emplace_back([this]() { worker_thread(); });
    }
  }

  ~ThreadPool() { finish(); }

  // Submit a job to be executed by the pool. This call blocks until a
  // worker thread is available to take the job.
  void submit_job(std::function<void()> &&job) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_job_finished.wait(lock, [this]() {
      return m_jobs.size() + m_running_jobs < m_thread_count;
    });
    m_jobs.emplace(std::forward<std::function<void()>>(job));
    lock.unlock();
    m_job_available.notify_one();
  }

  // Join all worker threads and clear any remaining jobs.
  void finish() {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_stop = true;
    }
    m_job_available.notify_all();
    for (std::thread &worker : m_workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    m_workers.clear();
    m_jobs = std::queue<std::function<void()>>();
  }

private:
  void worker_thread() {
    while (true) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_job_available.wait(lock,
                             [this]() { return m_stop || !m_jobs.empty(); });
        if (m_stop && m_jobs.empty()) {
          return;
        }
        job = std::move(m_jobs.front());
        m_jobs.pop();
        ++m_running_jobs;
      }
      job();
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        --m_running_jobs;
      }
      m_job_finished.notify_one();
    }
  }

  std::vector<std::thread> m_workers;
  std::queue<std::function<void()>> m_jobs;

  std::mutex m_mutex;
  std::condition_variable m_job_available;
  std::condition_variable m_job_finished;

  size_t m_running_jobs = 0;
  size_t m_thread_count;
  bool m_stop = false;
};