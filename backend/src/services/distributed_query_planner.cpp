#include "distributed_query_planner.h"
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>

namespace jadevectordb {

DistributedQueryPlanner::DistributedQueryPlanner(
    std::shared_ptr<ShardingService> sharding_service
) : sharding_service_(sharding_service),
    total_queries_planned_(0),
    total_planning_time_ms_(0),
    initialized_(false) {
    logger_ = logging::get_logger("DistributedQueryPlanner");
    recent_executions_.reserve(100);
}

DistributedQueryPlanner::~DistributedQueryPlanner() {
    if (initialized_) {
        shutdown();
    }
}

// ============================================================================
// Lifecycle Methods
// ============================================================================

Result<bool> DistributedQueryPlanner::initialize() {
    if (initialized_) {
        return create_error(ErrorCode::INVALID_STATE, "Query planner already initialized");
    }

    logger_->info("Initializing distributed query planner");

    if (!sharding_service_) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "ShardingService not provided");
    }

    initialized_ = true;
    logger_->info("Distributed query planner initialized successfully");
    return true;
}

Result<bool> DistributedQueryPlanner::shutdown() {
    if (!initialized_) {
        return true;
    }

    logger_->info("Shutting down distributed query planner");

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        recent_executions_.clear();
    }

    initialized_ = false;
    logger_->info("Distributed query planner shut down");
    return true;
}

// ============================================================================
// Query Planning
// ============================================================================

Result<QueryPlan> DistributedQueryPlanner::plan_query(
    const std::string& database_id,
    const std::vector<float>& query_vector,
    int top_k,
    const std::string& metric_type,
    float threshold,
    const std::unordered_map<std::string, std::string>& filters
) {
    auto start = std::chrono::steady_clock::now();

    logger_->debug("Planning query for database: " + database_id + ", top_k=" + std::to_string(top_k));

    // Create query plan
    QueryPlan plan;
    plan.query_id = generate_query_id();
    plan.database_id = database_id;
    plan.query_vector = query_vector;
    plan.top_k = top_k;
    plan.metric_type = metric_type;
    plan.threshold = threshold;
    plan.filters = filters;

    // Determine target shards
    auto shard_result = determine_target_shards(database_id, query_vector, filters);
    if (!shard_result.has_value()) {
        return tl::unexpected(shard_result.error());
    }

    std::vector<std::string> target_shards = shard_result.value();

    if (target_shards.empty()) {
        logger_->warn("No target shards found for database: " + database_id);
        return plan;  // Return empty plan
    }

    logger_->debug("Found " + std::to_string(target_shards.size()) + " target shards");

    // Map shards to workers
    auto worker_map_result = map_shards_to_workers(target_shards);
    if (!worker_map_result.has_value()) {
        return tl::unexpected(worker_map_result.error());
    }

    auto worker_map = worker_map_result.value();

    // Build shard targets
    for (const auto& shard_id : target_shards) {
        auto worker_it = worker_map.find(shard_id);
        if (worker_it == worker_map.end()) {
            logger_->warn("No worker found for shard: " + shard_id);
            continue;
        }

        QueryPlan::ShardTarget target;
        target.shard_id = shard_id;
        target.worker_id = worker_it->second;
        target.is_primary = true;  // TODO: Get from shard metadata
        target.timeout = config_.default_shard_timeout;
        target.priority = calculate_shard_priority(shard_id, target.is_primary, plan);

        plan.shard_targets.push_back(target);
    }

    // Select execution strategy
    plan.strategy = select_strategy(plan);

    // Set optimization hints
    plan.can_execute_parallel = can_parallelize(plan);
    plan.requires_global_ranking = (plan.shard_targets.size() > 1);
    plan.estimated_result_count = estimate_result_count(plan.shard_targets.size(), top_k);
    plan.estimated_latency = estimate_latency(plan.shard_targets.size(), top_k, plan.strategy);

    // Set failure handling
    plan.allow_partial_results = config_.allow_partial_results;
    plan.min_successful_shards = std::max(
        1,
        static_cast<int>(plan.shard_targets.size() * config_.min_shard_success_rate)
    );

    // Apply optimizations
    if (config_.enable_adaptive_strategy) {
        apply_adaptive_optimizations(plan);
    }

    if (plan.strategy == QueryPlan::ExecutionStrategy::SEQUENTIAL) {
        optimize_shard_order(plan);
    }

    auto end = std::chrono::steady_clock::now();
    auto planning_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    update_planning_stats(planning_time_ms);

    logger_->info("Query plan created: query_id=" + plan.query_id +
                  ", shards=" + std::to_string(plan.shard_targets.size()) +
                  ", strategy=" + std::to_string(static_cast<int>(plan.strategy)) +
                  ", planning_time=" + std::to_string(planning_time_ms) + "ms");

    return plan;
}

Result<QueryPlan> DistributedQueryPlanner::optimize_plan(const QueryPlan& plan) {
    logger_->debug("Optimizing query plan: " + plan.query_id);

    QueryPlan optimized = plan;

    // Re-evaluate execution strategy
    optimized.strategy = select_strategy(optimized);

    // Apply adaptive optimizations
    if (config_.enable_adaptive_strategy) {
        apply_adaptive_optimizations(optimized);
    }

    // Optimize shard ordering if sequential
    if (optimized.strategy == QueryPlan::ExecutionStrategy::SEQUENTIAL) {
        optimize_shard_order(optimized);
    }

    // Adjust timeouts based on historical data
    // TODO: Use historical latency data

    logger_->debug("Query plan optimized");
    return optimized;
}

Result<std::vector<std::string>> DistributedQueryPlanner::analyze_relevant_shards(
    const std::string& database_id,
    const std::vector<float>& query_vector,
    const std::unordered_map<std::string, std::string>& filters
) {
    return determine_target_shards(database_id, query_vector, filters);
}

QueryPlan::ExecutionStrategy DistributedQueryPlanner::select_strategy(const QueryPlan& plan) const {
    if (!config_.enable_parallel_execution) {
        return QueryPlan::ExecutionStrategy::SEQUENTIAL;
    }

    // If only one shard, no strategy needed
    if (plan.shard_targets.size() == 1) {
        return QueryPlan::ExecutionStrategy::PARALLEL_ALL;
    }

    // For small queries, always parallel
    if (plan.top_k <= config_.small_query_threshold) {
        return QueryPlan::ExecutionStrategy::PARALLEL_ALL;
    }

    // For very large queries, be more conservative
    if (plan.top_k >= config_.large_query_threshold) {
        if (plan.shard_targets.size() > config_.max_parallel_shards) {
            return QueryPlan::ExecutionStrategy::ADAPTIVE;
        }
    }

    // Default to parallel all
    return QueryPlan::ExecutionStrategy::PARALLEL_ALL;
}

// ============================================================================
// Execution Support
// ============================================================================

void DistributedQueryPlanner::record_execution_stats(const QueryExecutionStats& stats) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    recent_executions_.push_back(stats);

    // Keep only last 100
    if (recent_executions_.size() > 100) {
        recent_executions_.erase(recent_executions_.begin());
    }

    logger_->debug("Recorded execution stats for query: " + stats.query_id +
                   ", total_time=" + std::to_string(stats.total_time_ms) + "ms");
}

std::vector<QueryExecutionStats> DistributedQueryPlanner::get_recent_stats(int count) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    int start_idx = std::max(0, static_cast<int>(recent_executions_.size()) - count);
    return std::vector<QueryExecutionStats>(
        recent_executions_.begin() + start_idx,
        recent_executions_.end()
    );
}

DistributedQueryPlanner::AggregateStats DistributedQueryPlanner::get_aggregate_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    AggregateStats agg;
    agg.total_queries = total_queries_planned_;

    if (total_queries_planned_ > 0) {
        agg.avg_planning_time_ms = static_cast<double>(total_planning_time_ms_) / total_queries_planned_;
    }

    if (!recent_executions_.empty()) {
        int64_t total_exec_time = 0;
        int64_t total_merge_time = 0;
        int64_t total_total_time = 0;
        int64_t total_shards = 0;
        int64_t success_shards = 0;
        int64_t failed_shards = 0;

        for (const auto& stats : recent_executions_) {
            total_exec_time += stats.execution_time_ms;
            total_merge_time += stats.merge_time_ms;
            total_total_time += stats.total_time_ms;
            total_shards += stats.total_shards_queried;
            success_shards += stats.successful_shards;
            failed_shards += stats.failed_shards;
        }

        size_t count = recent_executions_.size();
        agg.avg_execution_time_ms = static_cast<double>(total_exec_time) / count;
        agg.avg_merge_time_ms = static_cast<double>(total_merge_time) / count;
        agg.avg_total_time_ms = static_cast<double>(total_total_time) / count;
        agg.total_shards_queried = total_shards;
        agg.successful_shards = success_shards;
        agg.failed_shards = failed_shards;

        if (total_shards > 0) {
            agg.avg_shard_success_rate = static_cast<double>(success_shards) / total_shards;
        }
    }

    return agg;
}

// ============================================================================
// Configuration
// ============================================================================

Result<bool> DistributedQueryPlanner::update_config(const PlannerConfig& new_config) {
    // Validate config
    if (new_config.min_shard_success_rate < 0.0 || new_config.min_shard_success_rate > 1.0) {
        return create_error(ErrorCode::INVALID_ARGUMENT,
                          "min_shard_success_rate must be between 0.0 and 1.0");
    }

    if (new_config.max_parallel_shards < 1) {
        return create_error(ErrorCode::INVALID_ARGUMENT,
                          "max_parallel_shards must be at least 1");
    }

    config_ = new_config;

    logger_->info("Query planner configuration updated");
    return true;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

Result<std::vector<std::string>> DistributedQueryPlanner::determine_target_shards(
    const std::string& database_id,
    const std::vector<float>& query_vector,
    const std::unordered_map<std::string, std::string>& filters
) {
    // Get all shards for the database from sharding service
    auto shards_result = sharding_service_->get_shards_for_database(database_id);

    if (!shards_result.has_value()) {
        logger_->warn("Failed to get shards for database: " + database_id);
        // Return all shards as fallback
        return std::vector<std::string>();
    }

    std::vector<std::string> all_shards = shards_result.value();

    // If no filters, all shards are relevant
    if (filters.empty()) {
        logger_->debug("No filters, all " + std::to_string(all_shards.size()) + " shards are relevant");
        return all_shards;
    }

    // TODO: Filter shards based on metadata filters
    // For now, return all shards
    return all_shards;
}

Result<std::unordered_map<std::string, std::string>> DistributedQueryPlanner::map_shards_to_workers(
    const std::vector<std::string>& shard_ids
) {
    std::unordered_map<std::string, std::string> mapping;

    for (const auto& shard_id : shard_ids) {
        // Get worker for shard from sharding service
        auto worker_result = sharding_service_->get_node_for_shard(shard_id);

        if (worker_result.has_value()) {
            mapping[shard_id] = worker_result.value();
        } else {
            logger_->warn("Failed to get worker for shard: " + shard_id);
        }
    }

    if (mapping.empty()) {
        return create_error(ErrorCode::NOT_FOUND, "No workers found for shards");
    }

    return mapping;
}

std::chrono::milliseconds DistributedQueryPlanner::estimate_latency(
    int shard_count,
    int top_k,
    QueryPlan::ExecutionStrategy strategy
) const {
    // Base latency per shard (in ms)
    int64_t base_latency_ms = 10;

    // Add latency based on top_k (more results = more processing)
    int64_t topk_latency_ms = top_k / 10;  // ~1ms per 10 results

    int64_t total_latency_ms = 0;

    switch (strategy) {
        case QueryPlan::ExecutionStrategy::PARALLEL_ALL:
        case QueryPlan::ExecutionStrategy::PARALLEL_PRIMARY_ONLY:
            // Parallel: latency is max of all shards (assume they're similar)
            total_latency_ms = base_latency_ms + topk_latency_ms;
            break;

        case QueryPlan::ExecutionStrategy::SEQUENTIAL:
            // Sequential: sum of all shards
            total_latency_ms = (base_latency_ms + topk_latency_ms) * shard_count;
            break;

        case QueryPlan::ExecutionStrategy::ADAPTIVE:
            // Adaptive: somewhere in between
            total_latency_ms = (base_latency_ms + topk_latency_ms) * (shard_count / 2);
            break;
    }

    // Add merge overhead
    int64_t merge_latency_ms = (shard_count * top_k) / 100;  // ~1ms per 100 results to merge

    return std::chrono::milliseconds(total_latency_ms + merge_latency_ms);
}

int DistributedQueryPlanner::estimate_result_count(int shard_count, int top_k) const {
    // In best case, we get top_k results
    // In worst case with duplicates, we get less
    // Estimate: 80% of top_k on average
    return std::min(top_k, static_cast<int>(top_k * 0.8));
}

int DistributedQueryPlanner::calculate_shard_priority(
    const std::string& shard_id,
    bool is_primary,
    const QueryPlan& plan
) const {
    // Lower priority number = higher priority
    int priority = 0;

    // Primary shards have higher priority
    if (!is_primary) {
        priority += 100;
    }

    // TODO: Add historical latency-based priority
    // Faster shards should have higher priority

    return priority;
}

bool DistributedQueryPlanner::can_parallelize(const QueryPlan& plan) const {
    if (!config_.enable_parallel_execution) {
        return false;
    }

    // Single shard doesn't need parallelization
    if (plan.shard_targets.size() <= 1) {
        return false;
    }

    // Check if we're within parallel limits
    if (plan.shard_targets.size() > static_cast<size_t>(config_.max_parallel_shards)) {
        return false;
    }

    return true;
}

void DistributedQueryPlanner::optimize_shard_order(QueryPlan& plan) const {
    // Sort shards by priority (lower = higher priority)
    std::sort(plan.shard_targets.begin(), plan.shard_targets.end(),
              [](const QueryPlan::ShardTarget& a, const QueryPlan::ShardTarget& b) {
                  return a.priority < b.priority;
              });

    logger_->debug("Optimized shard order for sequential execution");
}

void DistributedQueryPlanner::apply_adaptive_optimizations(QueryPlan& plan) const {
    // Get recent execution stats
    std::lock_guard<std::mutex> lock(stats_mutex_);

    if (recent_executions_.empty()) {
        return;
    }

    // Calculate average shard success rate
    int64_t total_shards = 0;
    int64_t successful_shards = 0;

    for (const auto& stats : recent_executions_) {
        total_shards += stats.total_shards_queried;
        successful_shards += stats.successful_shards;
    }

    if (total_shards > 0) {
        double success_rate = static_cast<double>(successful_shards) / total_shards;

        // If success rate is low, adjust strategy
        if (success_rate < 0.7) {
            logger_->warn("Low shard success rate: " + std::to_string(success_rate) +
                         ", adjusting strategy");

            // Be more conservative with parallelism
            if (plan.strategy == QueryPlan::ExecutionStrategy::PARALLEL_ALL) {
                plan.strategy = QueryPlan::ExecutionStrategy::ADAPTIVE;
            }

            // Increase timeouts
            for (auto& target : plan.shard_targets) {
                target.timeout = std::chrono::milliseconds(
                    static_cast<int64_t>(target.timeout.count() * 1.5)
                );
            }
        }
    }
}

std::string DistributedQueryPlanner::generate_query_id() const {
    static std::atomic<int64_t> counter{0};

    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();

    std::stringstream ss;
    ss << "q_" << timestamp << "_" << counter++;
    return ss.str();
}

void DistributedQueryPlanner::update_planning_stats(int64_t planning_time_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    total_queries_planned_++;
    total_planning_time_ms_ += planning_time_ms;
}

} // namespace jadevectordb
