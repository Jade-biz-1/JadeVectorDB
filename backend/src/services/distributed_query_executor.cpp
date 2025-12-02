#include "distributed_query_executor.h"
#include <algorithm>
#include <queue>
#include <numeric>

namespace jadevectordb {

// ============================================================================
// ThreadPool Implementation
// ============================================================================

ThreadPool::ThreadPool(size_t num_threads) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_thread, this);
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty()) {
                return;
            }

            if (!tasks_.empty()) {
                task = std::move(tasks_.front());
                tasks_.pop();
            }
        }

        if (task) {
            active_tasks_++;
            task();
            active_tasks_--;
        }
    }
}

void ThreadPool::wait_all() {
    while (active_tasks_ > 0 || !tasks_.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void ThreadPool::shutdown() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }

    condition_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    workers_.clear();
}

// ============================================================================
// DistributedQueryExecutor Implementation
// ============================================================================

DistributedQueryExecutor::DistributedQueryExecutor(
    std::shared_ptr<DistributedQueryPlanner> planner,
    std::shared_ptr<DistributedMasterClient> master_client
) : planner_(planner),
    master_client_(master_client),
    total_queries_executed_(0),
    successful_queries_(0),
    failed_queries_(0),
    cancelled_queries_(0),
    partial_queries_(0),
    initialized_(false),
    shutdown_(false) {
    logger_ = logging::get_logger("DistributedQueryExecutor");
}

DistributedQueryExecutor::~DistributedQueryExecutor() {
    if (initialized_ && !shutdown_) {
        shutdown_executor();
    }
}

// ============================================================================
// Lifecycle Methods
// ============================================================================

Result<bool> DistributedQueryExecutor::initialize() {
    if (initialized_) {
        return create_error(ErrorCode::INVALID_STATE, "Query executor already initialized");
    }

    logger_->info("Initializing distributed query executor");
    logger_->info("  Thread pool size: " + std::to_string(config_.thread_pool_size));
    logger_->info("  Global timeout: " + std::to_string(config_.global_query_timeout.count()) + "ms");

    if (!planner_) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "Query planner not provided");
    }

    if (!master_client_) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "Master client not provided");
    }

    // Initialize thread pool
    thread_pool_ = std::make_unique<ThreadPool>(config_.thread_pool_size);

    initialized_ = true;
    logger_->info("Distributed query executor initialized successfully");
    return true;
}

Result<bool> DistributedQueryExecutor::shutdown_executor() {
    if (shutdown_) {
        return true;
    }

    logger_->info("Shutting down distributed query executor");

    // Cancel all active queries
    {
        std::lock_guard<std::mutex> lock(active_queries_mutex_);
        for (auto& [query_id, query] : active_queries_) {
            query->cancelled = true;
            logger_->info("Cancelled active query: " + query_id);
        }
    }

    // Shutdown thread pool
    if (thread_pool_) {
        thread_pool_->wait_all();
        thread_pool_->shutdown();
        thread_pool_.reset();
    }

    shutdown_ = true;
    logger_->info("Distributed query executor shut down");
    return true;
}

// ============================================================================
// Query Execution
// ============================================================================

Result<SearchResults> DistributedQueryExecutor::execute_query(
    const std::string& database_id,
    const std::vector<float>& query_vector,
    int top_k,
    const std::string& metric_type,
    float threshold,
    const std::unordered_map<std::string, std::string>& filters
) {
    if (!initialized_ || shutdown_) {
        return create_error(ErrorCode::INVALID_STATE, "Query executor not available");
    }

    logger_->info("Executing distributed query: db=" + database_id + ", top_k=" + std::to_string(top_k));

    // Create query plan
    auto plan_result = planner_->plan_query(database_id, query_vector, top_k, metric_type, threshold, filters);
    if (!plan_result.has_value()) {
        record_query_result(false, false, false);
        return tl::unexpected(plan_result.error());
    }

    // Execute plan
    return execute_plan(plan_result.value());
}

Result<SearchResults> DistributedQueryExecutor::execute_plan(const QueryPlan& plan) {
    auto start = std::chrono::steady_clock::now();

    logger_->debug("Executing query plan: " + plan.query_id +
                   ", shards=" + std::to_string(plan.shard_targets.size()) +
                   ", strategy=" + std::to_string(static_cast<int>(plan.strategy)));

    // Register query as active
    auto active_query = register_query(plan);

    // Execute based on strategy
    Result<std::vector<ShardQueryResult>> shard_results;

    switch (plan.strategy) {
        case QueryPlan::ExecutionStrategy::PARALLEL_ALL:
        case QueryPlan::ExecutionStrategy::PARALLEL_PRIMARY_ONLY:
            shard_results = execute_parallel(plan, active_query);
            break;

        case QueryPlan::ExecutionStrategy::SEQUENTIAL:
            shard_results = execute_sequential(plan, active_query);
            break;

        case QueryPlan::ExecutionStrategy::ADAPTIVE:
            shard_results = execute_adaptive(plan, active_query);
            break;

        default:
            unregister_query(plan.query_id);
            return create_error(ErrorCode::INVALID_ARGUMENT, "Unknown execution strategy");
    }

    auto execution_end = std::chrono::steady_clock::now();
    auto execution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        execution_end - start).count();

    // Check if query was cancelled
    if (active_query->cancelled) {
        unregister_query(plan.query_id);
        record_query_result(false, true, false);
        return create_error(ErrorCode::CANCELLED, "Query was cancelled");
    }

    // Validate results
    if (!shard_results.has_value()) {
        unregister_query(plan.query_id);
        record_query_result(false, false, false);
        return tl::unexpected(shard_results.error());
    }

    auto validation_result = validate_shard_results(shard_results.value(), plan);
    if (!validation_result.has_value()) {
        unregister_query(plan.query_id);
        record_query_result(false, false, false);
        return tl::unexpected(validation_result.error());
    }

    // Merge results
    auto merged_results = merge_results(shard_results.value(), plan);

    auto merge_end = std::chrono::steady_clock::now();
    auto merge_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        merge_end - execution_end).count();

    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        merge_end - start).count();

    // Record statistics
    QueryExecutionStats stats;
    stats.query_id = plan.query_id;
    stats.execution_time_ms = execution_time_ms;
    stats.merge_time_ms = merge_time_ms;
    stats.total_time_ms = total_time_ms;
    stats.total_shards_queried = plan.shard_targets.size();

    for (const auto& sr : shard_results.value()) {
        if (sr.success) {
            stats.successful_shards++;
            stats.total_results_from_shards += sr.results.size();
            stats.per_shard_latency[sr.shard_id] = sr.execution_time_ms;
        } else {
            stats.failed_shards++;
        }
    }

    if (merged_results.has_value()) {
        stats.final_result_count = merged_results.value().results.size();
    }

    planner_->record_execution_stats(stats);

    // Determine if this was a partial success
    bool partial = (stats.failed_shards > 0 && stats.successful_shards > 0);

    unregister_query(plan.query_id);

    if (merged_results.has_value()) {
        record_query_result(true, false, partial);
        logger_->info("Query completed: " + plan.query_id +
                     ", total_time=" + std::to_string(total_time_ms) + "ms" +
                     ", results=" + std::to_string(stats.final_result_count));
    } else {
        record_query_result(false, false, false);
    }

    return merged_results;
}

Result<bool> DistributedQueryExecutor::cancel_query(const std::string& query_id) {
    std::lock_guard<std::mutex> lock(active_queries_mutex_);

    auto it = active_queries_.find(query_id);
    if (it == active_queries_.end()) {
        return create_error(ErrorCode::NOT_FOUND, "Query not found: " + query_id);
    }

    it->second->cancelled = true;
    logger_->info("Query cancelled: " + query_id);

    return true;
}

bool DistributedQueryExecutor::is_query_active(const std::string& query_id) const {
    std::lock_guard<std::mutex> lock(active_queries_mutex_);
    return active_queries_.find(query_id) != active_queries_.end();
}

std::vector<std::string> DistributedQueryExecutor::get_active_queries() const {
    std::lock_guard<std::mutex> lock(active_queries_mutex_);

    std::vector<std::string> queries;
    for (const auto& [query_id, _] : active_queries_) {
        queries.push_back(query_id);
    }

    return queries;
}

// ============================================================================
// Statistics
// ============================================================================

DistributedQueryExecutor::ExecutorStats DistributedQueryExecutor::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    ExecutorStats stats;
    stats.total_queries = total_queries_executed_;
    stats.successful_queries = successful_queries_;
    stats.failed_queries = failed_queries_;
    stats.cancelled_queries = cancelled_queries_;
    stats.partial_queries = partial_queries_;

    if (total_queries_executed_ > 0) {
        stats.success_rate = static_cast<double>(successful_queries_) / total_queries_executed_;
    } else {
        stats.success_rate = 0.0;
    }

    std::lock_guard<std::mutex> aq_lock(active_queries_mutex_);
    stats.active_queries = active_queries_.size();

    if (thread_pool_) {
        stats.thread_pool_active_tasks = thread_pool_->get_active_tasks();
    }

    return stats;
}

Result<bool> DistributedQueryExecutor::update_config(const ExecutorConfig& new_config) {
    if (new_config.thread_pool_size < 1) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "Thread pool size must be at least 1");
    }

    if (new_config.min_success_rate < 0.0 || new_config.min_success_rate > 1.0) {
        return create_error(ErrorCode::INVALID_ARGUMENT,
                          "min_success_rate must be between 0.0 and 1.0");
    }

    // Note: Changing thread pool size requires reinitialization
    config_ = new_config;

    logger_->info("Query executor configuration updated");
    return true;
}

// ============================================================================
// Execution Methods
// ============================================================================

Result<std::vector<ShardQueryResult>> DistributedQueryExecutor::execute_parallel(
    const QueryPlan& plan,
    std::shared_ptr<ActiveQuery> active_query
) {
    logger_->debug("Executing parallel query across " + std::to_string(plan.shard_targets.size()) + " shards");

    std::vector<std::future<ShardQueryResult>> futures;

    // Launch all shard queries in parallel
    for (const auto& target : plan.shard_targets) {
        if (!should_continue(active_query)) {
            break;
        }

        auto future = thread_pool_->enqueue([this, target, plan, active_query]() {
            return execute_shard_query(target, plan, active_query);
        });

        futures.push_back(std::move(future));
    }

    // Collect results
    std::vector<ShardQueryResult> results;
    for (auto& future : futures) {
        try {
            results.push_back(future.get());
        } catch (const std::exception& e) {
            logger_->error("Exception in shard query: " + std::string(e.what()));
        }
    }

    return results;
}

Result<std::vector<ShardQueryResult>> DistributedQueryExecutor::execute_sequential(
    const QueryPlan& plan,
    std::shared_ptr<ActiveQuery> active_query
) {
    logger_->debug("Executing sequential query across " + std::to_string(plan.shard_targets.size()) + " shards");

    std::vector<ShardQueryResult> results;

    for (const auto& target : plan.shard_targets) {
        if (!should_continue(active_query)) {
            break;
        }

        auto result = execute_shard_query(target, plan, active_query);
        results.push_back(result);
    }

    return results;
}

Result<std::vector<ShardQueryResult>> DistributedQueryExecutor::execute_adaptive(
    const QueryPlan& plan,
    std::shared_ptr<ActiveQuery> active_query
) {
    logger_->debug("Executing adaptive query across " + std::to_string(plan.shard_targets.size()) + " shards");

    // Start with parallel for first batch
    size_t batch_size = std::min(plan.shard_targets.size(), config_.thread_pool_size);
    std::vector<ShardQueryResult> results;

    for (size_t i = 0; i < plan.shard_targets.size(); i += batch_size) {
        if (!should_continue(active_query)) {
            break;
        }

        size_t end = std::min(i + batch_size, plan.shard_targets.size());
        std::vector<std::future<ShardQueryResult>> futures;

        for (size_t j = i; j < end; ++j) {
            auto future = thread_pool_->enqueue([this, target = plan.shard_targets[j], plan, active_query]() {
                return execute_shard_query(target, plan, active_query);
            });

            futures.push_back(std::move(future));
        }

        for (auto& future : futures) {
            try {
                results.push_back(future.get());
            } catch (const std::exception& e) {
                logger_->error("Exception in adaptive query: " + std::string(e.what()));
            }
        }
    }

    return results;
}

ShardQueryResult DistributedQueryExecutor::execute_shard_query(
    const QueryPlan::ShardTarget& target,
    const QueryPlan& plan,
    std::shared_ptr<ActiveQuery> active_query
) {
    ShardQueryResult result;
    result.shard_id = target.shard_id;
    result.worker_id = target.worker_id;

    if (!should_continue(active_query)) {
        result.success = false;
        result.error_message = "Query cancelled";
        return result;
    }

    logger_->debug("Executing query on shard: " + target.shard_id + " @ worker: " + target.worker_id);

    auto start = std::chrono::steady_clock::now();

    // Create search request for master client
    DistributedMasterClient::SearchRequest search_req;
    search_req.shard_id = target.shard_id;
    search_req.request_id = plan.query_id;
    search_req.query_vector = plan.query_vector;
    search_req.top_k = plan.top_k;
    search_req.metric_type = plan.metric_type;
    search_req.threshold = plan.threshold;
    search_req.filters = plan.filters;
    search_req.timeout = target.timeout;

    // Execute RPC call
    auto response = master_client_->execute_shard_search(target.worker_id, search_req);

    auto end = std::chrono::steady_clock::now();
    result.execution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (response.has_value()) {
        result.success = response.value().success;
        result.error_message = response.value().error_message;
        result.results = response.value().results;
        result.vectors_scanned = response.value().vectors_scanned;

        logger_->debug("Shard query succeeded: " + target.shard_id +
                      ", results=" + std::to_string(result.results.size()) +
                      ", time=" + std::to_string(result.execution_time_ms) + "ms");
    } else {
        result.success = false;
        result.error_message = response.error().message;

        logger_->warn("Shard query failed: " + target.shard_id + ", error=" + result.error_message);
    }

    return result;
}

Result<SearchResults> DistributedQueryExecutor::merge_results(
    const std::vector<ShardQueryResult>& shard_results,
    const QueryPlan& plan
) {
    logger_->debug("Merging results from " + std::to_string(shard_results.size()) + " shards");

    SearchResults merged;
    merged.database_id = plan.database_id;
    merged.query_vector = plan.query_vector;

    if (shard_results.empty()) {
        return merged;
    }

    // Collect all results with their scores
    std::vector<SearchResult> all_results;

    for (const auto& shard_result : shard_results) {
        if (shard_result.success) {
            for (const auto& result : shard_result.results) {
                all_results.push_back(result);
            }
        }
    }

    // Sort by score (descending)
    std::sort(all_results.begin(), all_results.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.score > b.score;
              });

    // Take top_k results
    size_t count = std::min(static_cast<size_t>(plan.top_k), all_results.size());
    merged.results.assign(all_results.begin(), all_results.begin() + count);

    logger_->debug("Merged " + std::to_string(merged.results.size()) + " results");

    return merged;
}

// ============================================================================
// Helper Methods
// ============================================================================

bool DistributedQueryExecutor::should_continue(std::shared_ptr<ActiveQuery> active_query) const {
    if (active_query->cancelled) {
        return false;
    }

    // Check timeout
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - active_query->start_time);

    if (elapsed > config_.global_query_timeout) {
        logger_->warn("Query timeout: " + active_query->query_id);
        return false;
    }

    return true;
}

Result<bool> DistributedQueryExecutor::validate_shard_results(
    const std::vector<ShardQueryResult>& shard_results,
    const QueryPlan& plan
) const {
    int successful_shards = 0;

    for (const auto& result : shard_results) {
        if (result.success) {
            successful_shards++;
        }
    }

    double success_rate = static_cast<double>(successful_shards) / shard_results.size();

    if (success_rate < config_.min_success_rate) {
        return create_error(ErrorCode::INTERNAL_ERROR,
                          "Insufficient successful shards: " + std::to_string(successful_shards) +
                          "/" + std::to_string(shard_results.size()));
    }

    if (successful_shards < plan.min_successful_shards) {
        return create_error(ErrorCode::INTERNAL_ERROR,
                          "Below minimum successful shards: " + std::to_string(successful_shards) +
                          "/" + std::to_string(plan.min_successful_shards));
    }

    return true;
}

std::shared_ptr<DistributedQueryExecutor::ActiveQuery> DistributedQueryExecutor::register_query(
    const QueryPlan& plan
) {
    auto active_query = std::make_shared<ActiveQuery>();
    active_query->query_id = plan.query_id;
    active_query->start_time = std::chrono::steady_clock::now();
    active_query->plan = plan;

    std::lock_guard<std::mutex> lock(active_queries_mutex_);
    active_queries_[plan.query_id] = active_query;

    logger_->debug("Registered active query: " + plan.query_id);

    return active_query;
}

void DistributedQueryExecutor::unregister_query(const std::string& query_id) {
    std::lock_guard<std::mutex> lock(active_queries_mutex_);
    active_queries_.erase(query_id);

    logger_->debug("Unregistered query: " + query_id);
}

void DistributedQueryExecutor::record_query_result(bool success, bool cancelled, bool partial) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    total_queries_executed_++;

    if (cancelled) {
        cancelled_queries_++;
    } else if (success) {
        successful_queries_++;
        if (partial) {
            partial_queries_++;
        }
    } else {
        failed_queries_++;
    }
}

} // namespace jadevectordb
