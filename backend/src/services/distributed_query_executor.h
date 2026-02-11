#ifndef JADEVECTORDB_DISTRIBUTED_QUERY_EXECUTOR_H
#define JADEVECTORDB_DISTRIBUTED_QUERY_EXECUTOR_H

#include "lib/error_handling.h"
#include "lib/logging.h"
#include "models/vector.h"
#include "distributed_query_planner.h"
#include "api/grpc/distributed_master_client.h"
#include "similarity_search.h"
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <future>
#include <atomic>
#include <queue>
#include <condition_variable>

namespace jadevectordb {

/**
 * @brief Aggregated search results structure
 */
struct SearchResults {
    std::vector<SearchResult> results;
    int64_t total_time_ms{0};
    int64_t total_vectors_scanned{0};
    int shards_queried{0};
    bool success{true};
    std::string error_message;

    SearchResults() = default;
};

/**
 * @brief Result from a single shard query
 */
struct ShardQueryResult {
    std::string shard_id;
    std::string worker_id;
    bool success;
    std::string error_message;
    std::vector<SearchResult> results;
    int64_t execution_time_ms;
    int32_t vectors_scanned;

    ShardQueryResult() : success(false), execution_time_ms(0), vectors_scanned(0) {}
};

/**
 * @brief Thread pool for parallel query execution
 */
class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
    std::atomic<int> active_tasks_{0};

public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    // Enqueue a task
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>;

    // Get number of active tasks
    int get_active_tasks() const { return active_tasks_.load(); }

    // Wait for all tasks to complete
    void wait_all();

    // Shutdown the pool
    void shutdown();

private:
    void worker_thread();
};

/**
 * @brief Executes distributed queries across multiple worker nodes
 *
 * Features:
 * - Parallel query execution with thread pool
 * - Async RPC calls to workers
 * - Query cancellation support
 * - Partial failure handling
 * - Result aggregation and ranking
 * - Timeout management
 */
class DistributedQueryExecutor {
private:
    std::shared_ptr<logging::Logger> logger_;
    std::shared_ptr<DistributedQueryPlanner> planner_;
    std::shared_ptr<DistributedMasterClient> master_client_;

    // Configuration
    struct ExecutorConfig {
        size_t thread_pool_size{10};
        bool enable_query_cancellation{true};
        bool enable_partial_results{true};
        double min_success_rate{0.5};  // At least 50% of shards must succeed
        std::chrono::milliseconds global_query_timeout{10000};
        int max_concurrent_queries{100};

        ExecutorConfig() = default;
    };

    ExecutorConfig config_;

    // Thread pool for parallel execution
    std::unique_ptr<ThreadPool> thread_pool_;

    // Active queries tracking
    struct ActiveQuery {
        std::string query_id;
        std::atomic<bool> cancelled{false};
        std::chrono::steady_clock::time_point start_time;
        QueryPlan plan;
    };

    mutable std::mutex active_queries_mutex_;
    std::unordered_map<std::string, std::shared_ptr<ActiveQuery>> active_queries_;

    // Statistics
    mutable std::mutex stats_mutex_;
    int64_t total_queries_executed_;
    int64_t successful_queries_;
    int64_t failed_queries_;
    int64_t cancelled_queries_;
    int64_t partial_queries_;  // Succeeded with some shard failures

    bool initialized_;
    bool shutdown_;

public:
    explicit DistributedQueryExecutor(
        std::shared_ptr<DistributedQueryPlanner> planner,
        std::shared_ptr<DistributedMasterClient> master_client
    );

    ~DistributedQueryExecutor();

    // ===== Lifecycle =====

    Result<bool> initialize();
    Result<bool> shutdown_executor();
    bool is_initialized() const { return initialized_; }

    // ===== Query Execution =====

    /**
     * @brief Execute a distributed query
     * @param database_id Target database
     * @param query_vector Query vector
     * @param top_k Number of results
     * @param metric_type Distance metric
     * @param threshold Score threshold
     * @param filters Metadata filters
     * @return Aggregated search results
     */
    Result<SearchResults> execute_query(
        const std::string& database_id,
        const std::vector<float>& query_vector,
        int top_k,
        const std::string& metric_type = "cosine",
        float threshold = 0.0f,
        const std::unordered_map<std::string, std::string>& filters = {}
    );

    /**
     * @brief Execute a pre-planned query
     * @param plan Query execution plan
     * @return Aggregated search results
     */
    Result<SearchResults> execute_plan(const QueryPlan& plan);

    /**
     * @brief Cancel a running query
     * @param query_id Query ID to cancel
     * @return True if cancelled successfully
     */
    Result<bool> cancel_query(const std::string& query_id);

    /**
     * @brief Check if a query is running
     * @param query_id Query ID
     * @return True if query is active
     */
    bool is_query_active(const std::string& query_id) const;

    /**
     * @brief Get list of active query IDs
     * @return Vector of active query IDs
     */
    std::vector<std::string> get_active_queries() const;

    // ===== Statistics =====

    struct ExecutorStats {
        int64_t total_queries;
        int64_t successful_queries;
        int64_t failed_queries;
        int64_t cancelled_queries;
        int64_t partial_queries;
        double success_rate;
        int active_queries;
        int thread_pool_active_tasks;
    };

    ExecutorStats get_statistics() const;

    // ===== Configuration =====

    ExecutorConfig get_config() const { return config_; }
    Result<bool> update_config(const ExecutorConfig& new_config);

private:
    // Execution methods

    /**
     * @brief Execute query in parallel across all shards
     */
    Result<std::vector<ShardQueryResult>> execute_parallel(
        const QueryPlan& plan,
        std::shared_ptr<ActiveQuery> active_query
    );

    /**
     * @brief Execute query sequentially across shards
     */
    Result<std::vector<ShardQueryResult>> execute_sequential(
        const QueryPlan& plan,
        std::shared_ptr<ActiveQuery> active_query
    );

    /**
     * @brief Execute query with adaptive strategy
     */
    Result<std::vector<ShardQueryResult>> execute_adaptive(
        const QueryPlan& plan,
        std::shared_ptr<ActiveQuery> active_query
    );

    /**
     * @brief Execute single shard query
     */
    ShardQueryResult execute_shard_query(
        const QueryPlan::ShardTarget& target,
        const QueryPlan& plan,
        std::shared_ptr<ActiveQuery> active_query
    );

    /**
     * @brief Merge results from multiple shards
     */
    Result<SearchResults> merge_results(
        const std::vector<ShardQueryResult>& shard_results,
        const QueryPlan& plan
    );

    /**
     * @brief Check if query should continue or be cancelled
     */
    bool should_continue(std::shared_ptr<ActiveQuery> active_query) const;

    /**
     * @brief Validate shard results meet minimum requirements
     */
    Result<bool> validate_shard_results(
        const std::vector<ShardQueryResult>& shard_results,
        const QueryPlan& plan
    ) const;

    // Active query management
    std::shared_ptr<ActiveQuery> register_query(const QueryPlan& plan);
    void unregister_query(const std::string& query_id);

    // Statistics
    void record_query_result(bool success, bool cancelled, bool partial);
};

// ============================================================================
// ThreadPool Template Implementation
// ============================================================================

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        if (stop_) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }

        tasks_.emplace([task]() { (*task)(); });
    }

    condition_.notify_one();
    return res;
}

} // namespace jadevectordb

#endif // JADEVECTORDB_DISTRIBUTED_QUERY_EXECUTOR_H
