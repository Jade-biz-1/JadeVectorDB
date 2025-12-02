#ifndef JADEVECTORDB_DISTRIBUTED_QUERY_PLANNER_H
#define JADEVECTORDB_DISTRIBUTED_QUERY_PLANNER_H

#include "lib/error_handling.h"
#include "lib/logging.h"
#include "models/vector.h"
#include "models/database.h"
#include "sharding_service.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

namespace jadevectordb {

// Forward declarations
class ShardingService;

/**
 * @brief Query execution plan for distributed queries
 */
struct QueryPlan {
    std::string query_id;
    std::string database_id;

    // Query details
    std::vector<float> query_vector;
    int top_k;
    std::string metric_type;
    float threshold;
    std::unordered_map<std::string, std::string> filters;

    // Execution plan
    struct ShardTarget {
        std::string shard_id;
        std::string worker_id;
        bool is_primary;
        int priority;  // Execution priority (lower = higher priority)
        std::chrono::milliseconds timeout;
    };

    std::vector<ShardTarget> shard_targets;

    // Optimization hints
    bool can_execute_parallel;
    bool requires_global_ranking;
    int estimated_result_count;
    std::chrono::milliseconds estimated_latency;

    // Execution strategy
    enum class ExecutionStrategy {
        PARALLEL_ALL,           // Query all shards in parallel
        PARALLEL_PRIMARY_ONLY,  // Query only primary shards
        SEQUENTIAL,             // Query shards sequentially
        ADAPTIVE               // Adapt based on load/latency
    };

    ExecutionStrategy strategy;

    // Failure handling
    bool allow_partial_results;
    int min_successful_shards;  // Minimum shards that must succeed

    QueryPlan() :
        top_k(10),
        threshold(0.0f),
        can_execute_parallel(true),
        requires_global_ranking(true),
        estimated_result_count(0),
        estimated_latency(std::chrono::milliseconds(100)),
        strategy(ExecutionStrategy::PARALLEL_ALL),
        allow_partial_results(true),
        min_successful_shards(1) {}
};

/**
 * @brief Statistics about query execution
 */
struct QueryExecutionStats {
    std::string query_id;
    int64_t planning_time_ms;
    int64_t execution_time_ms;
    int64_t merge_time_ms;
    int64_t total_time_ms;

    int total_shards_queried;
    int successful_shards;
    int failed_shards;
    int total_results_from_shards;
    int final_result_count;

    std::unordered_map<std::string, int64_t> per_shard_latency;

    QueryExecutionStats() :
        planning_time_ms(0),
        execution_time_ms(0),
        merge_time_ms(0),
        total_time_ms(0),
        total_shards_queried(0),
        successful_shards(0),
        failed_shards(0),
        total_results_from_shards(0),
        final_result_count(0) {}
};

/**
 * @brief Query planner for distributed vector search
 *
 * Responsible for:
 * - Analyzing queries to determine relevant shards
 * - Generating optimal execution plans
 * - Optimizing for parallelism and latency
 * - Handling shard failures and fallbacks
 */
class DistributedQueryPlanner {
private:
    std::shared_ptr<logging::Logger> logger_;
    std::shared_ptr<ShardingService> sharding_service_;

    // Configuration
    struct PlannerConfig {
        // Query optimization
        bool enable_parallel_execution{true};
        bool enable_adaptive_strategy{true};
        int max_parallel_shards{10};

        // Timeout configuration
        std::chrono::milliseconds default_shard_timeout{2000};
        std::chrono::milliseconds max_query_timeout{10000};

        // Failure handling
        bool allow_partial_results{true};
        double min_shard_success_rate{0.5};  // At least 50% must succeed

        // Optimization thresholds
        int small_query_threshold{10};  // top_k <= this is considered small
        int large_query_threshold{100}; // top_k >= this is considered large

        PlannerConfig() = default;
    };

    PlannerConfig config_;

    // Statistics
    mutable std::mutex stats_mutex_;
    int64_t total_queries_planned_;
    int64_t total_planning_time_ms_;
    std::vector<QueryExecutionStats> recent_executions_;  // Keep last 100

    bool initialized_;

public:
    explicit DistributedQueryPlanner(std::shared_ptr<ShardingService> sharding_service);
    ~DistributedQueryPlanner();

    // ===== Lifecycle =====

    Result<bool> initialize();
    Result<bool> shutdown();
    bool is_initialized() const { return initialized_; }

    // ===== Query Planning =====

    /**
     * @brief Create execution plan for a distributed query
     * @param database_id Target database
     * @param query_vector Query vector
     * @param top_k Number of results
     * @param metric_type Distance metric (cosine, euclidean, etc.)
     * @param threshold Score threshold
     * @param filters Metadata filters
     * @return Query execution plan
     */
    Result<QueryPlan> plan_query(
        const std::string& database_id,
        const std::vector<float>& query_vector,
        int top_k,
        const std::string& metric_type = "cosine",
        float threshold = 0.0f,
        const std::unordered_map<std::string, std::string>& filters = {}
    );

    /**
     * @brief Optimize existing query plan
     * @param plan Query plan to optimize
     * @return Optimized query plan
     */
    Result<QueryPlan> optimize_plan(const QueryPlan& plan);

    /**
     * @brief Analyze query to determine relevant shards
     * @param database_id Target database
     * @param query_vector Query vector
     * @param filters Metadata filters
     * @return List of relevant shard IDs
     */
    Result<std::vector<std::string>> analyze_relevant_shards(
        const std::string& database_id,
        const std::vector<float>& query_vector,
        const std::unordered_map<std::string, std::string>& filters = {}
    );

    /**
     * @brief Select execution strategy based on query characteristics
     * @param plan Query plan
     * @return Optimal execution strategy
     */
    QueryPlan::ExecutionStrategy select_strategy(const QueryPlan& plan) const;

    // ===== Execution Support =====

    /**
     * @brief Record query execution statistics
     * @param stats Execution statistics
     */
    void record_execution_stats(const QueryExecutionStats& stats);

    /**
     * @brief Get recent execution statistics
     * @param count Number of recent executions to return
     * @return Recent execution statistics
     */
    std::vector<QueryExecutionStats> get_recent_stats(int count = 10) const;

    /**
     * @brief Get aggregate statistics
     */
    struct AggregateStats {
        int64_t total_queries;
        double avg_planning_time_ms;
        double avg_execution_time_ms;
        double avg_merge_time_ms;
        double avg_total_time_ms;
        double avg_shard_success_rate;
        int64_t total_shards_queried;
        int64_t successful_shards;
        int64_t failed_shards;
    };

    AggregateStats get_aggregate_stats() const;

    // ===== Configuration =====

    PlannerConfig get_config() const { return config_; }
    Result<bool> update_config(const PlannerConfig& new_config);

private:
    // Planning helpers

    /**
     * @brief Determine which shards contain relevant data
     */
    Result<std::vector<std::string>> determine_target_shards(
        const std::string& database_id,
        const std::vector<float>& query_vector,
        const std::unordered_map<std::string, std::string>& filters
    );

    /**
     * @brief Map shards to worker nodes
     */
    Result<std::unordered_map<std::string, std::string>> map_shards_to_workers(
        const std::vector<std::string>& shard_ids
    );

    /**
     * @brief Estimate query latency
     */
    std::chrono::milliseconds estimate_latency(
        int shard_count,
        int top_k,
        QueryPlan::ExecutionStrategy strategy
    ) const;

    /**
     * @brief Estimate result count
     */
    int estimate_result_count(
        int shard_count,
        int top_k
    ) const;

    /**
     * @brief Calculate priority for shard execution
     */
    int calculate_shard_priority(
        const std::string& shard_id,
        bool is_primary,
        const QueryPlan& plan
    ) const;

    /**
     * @brief Check if query can benefit from parallelism
     */
    bool can_parallelize(const QueryPlan& plan) const;

    /**
     * @brief Optimize shard ordering for sequential execution
     */
    void optimize_shard_order(QueryPlan& plan) const;

    /**
     * @brief Apply adaptive optimizations based on historical data
     */
    void apply_adaptive_optimizations(QueryPlan& plan) const;

    /**
     * @brief Generate unique query ID
     */
    std::string generate_query_id() const;

    // Statistics helpers
    void update_planning_stats(int64_t planning_time_ms);
};

} // namespace jadevectordb

#endif // JADEVECTORDB_DISTRIBUTED_QUERY_PLANNER_H
