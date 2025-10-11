#ifndef JADEVECTORDB_THREAD_POOL_H
#define JADEVECTORDB_THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

namespace jadevectordb {

class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
    std::atomic<size_t> active_tasks_;

public:
    explicit ThreadPool(size_t pool_size);
    
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    ~ThreadPool();
    
    size_t get_pool_size() const { return workers_.size(); }
    size_t get_active_tasks() const { return active_tasks_.load(); }
    
private:
    void worker_loop();
};

// Implementation of the template method
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Don't allow enqueueing after stopping the pool
        if (stop_) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        
        tasks_.emplace([task](){ (*task)(); });
        active_tasks_++;
    }
    
    condition_.notify_one();
    return res;
}

} // namespace jadevectordb

#endif // JADEVECTORDB_THREAD_POOL_H