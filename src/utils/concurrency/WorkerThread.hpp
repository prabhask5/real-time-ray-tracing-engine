#pragma once

#include "ThreadTypes.hpp"
#include "WorkStealingDeque.hpp"
#include <thread>

// Worker thread structure wrapper for regular std::thread.
//
// NOTE: This alignas(64) keyword forces this variable to be aligned on a
// 64-byte boundary (a typical cache line size).
struct alignas(64) WorkerThread {
  std::thread thread;
  WorkStealingDeque<Task> local_queue;
  std::atomic<bool> active{false};

  WorkerThread() = default;
  WorkerThread(const WorkerThread &) = delete;
  WorkerThread &operator=(const WorkerThread &) = delete;
  WorkerThread(WorkerThread &&) = delete;
  WorkerThread &operator=(WorkerThread &&) = delete;
};