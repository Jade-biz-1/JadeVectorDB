#include "thread_pool.h"

namespace jadevectordb {

ThreadPool::ThreadPool(size_t pool_size) 
    : stop_(false), active_tasks_(0) {
    for (size_t i = 0; i < pool_size; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            
            if (stop_ && tasks_.empty()) {
                return;
            }
            
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        
        task();
        active_tasks_--;
    }
}

ThreadPool::~ThreadPool() {
    stop_ = true;
    condition_.notify_all();
    
    for (std::thread &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

} // namespace jadevectordb