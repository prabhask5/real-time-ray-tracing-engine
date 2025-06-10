#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
  ThreadPool(size_t thread_count)
      : m_thread_count(thread_count), m_stop(true), m_running_jobs(0) {}

  ~ThreadPool() { finish(); }

  void start() {
    if (!m_stop)
      return;

    m_stop = false;

    for (size_t i = 0; i < m_thread_count; ++i) {
      m_workers.emplace_back([this] {
        while (true) {
          std::function<void()> job;

          {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_job_available.wait(lock,
                                 [this] { return m_stop || !m_jobs.empty(); });

            if (m_stop && m_jobs.empty())
              return;

            job = std::move(m_jobs.front());
            m_jobs.pop();
            ++m_running_jobs;
          }

          job();

          {
            std::unique_lock<std::mutex> lock(m_mutex);
            --m_running_jobs;
            if (m_jobs.empty() && m_running_jobs == 0) {
              m_all_done.notify_one();
            }
          }
        }
      });
    }
  }

  void submit_job(std::function<void()> job) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_jobs.push(std::move(job));
    m_job_available.notify_one();
  }

  void finish() {
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      m_all_done.wait(lock,
                      [this] { return m_jobs.empty() && m_running_jobs == 0; });
      m_stop = true;
    }

    m_job_available.notify_all();
    for (auto &t : m_workers) {
      if (t.joinable())
        t.join();
    }
    m_workers.clear();
  }

private:
  std::vector<std::thread> m_workers;
  std::queue<std::function<void()>> m_jobs;

  std::mutex m_mutex;
  std::condition_variable m_job_available;
  std::condition_variable m_all_done;

  std::atomic<bool> m_stop;
  size_t m_thread_count;
  size_t m_running_jobs;
};
