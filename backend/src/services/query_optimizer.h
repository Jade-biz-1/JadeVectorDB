#ifndef JADEVECTORDB_QUERY_OPTIMIZER_H
#define JADEVECTORDB_QUERY_OPTIMIZER_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <mutex>

namespace jadevectordb {

// Forward declarations
struct Vector;
struct SearchParams;

// Enum for index types
enum class IndexType {
    FLAT,      // Brute force scan
    HNSW,      // Hierarchical Navigable Small World
    IVF,       // Inverted File
    LSH,       // Locality Sensitive Hashing
    COMPOSITE  // Composite index
};

// Structure for index statistics
struct IndexStats {
    IndexType type;
    size_t vector_count;
    size_t dimension;
    double avg_query_time_ms;
    double build_time_ms;
    size_t memory_bytes;
    double recall_rate;  // For approximate indices
    std::chrono::system_clock::time_point last_updated;
};

// Structure for query optimization plan
struct QueryOptimizationPlan {
    std::string plan_id;
    IndexType selected_index;
    bool use_filter_pushdown;
    bool use_early_termination;
    std::vector<std::string> filter_order;  // Order to apply filters
    double estimated_cost;
    double estimated_time_ms;
    std::string reasoning;  // Why this plan was chosen
};

// Structure for query statistics
struct QueryStats {
    size_t total_queries = 0;
    double avg_latency_ms = 0.0;
    size_t vectors_scanned = 0;
    size_t results_returned = 0;
    std::chrono::system_clock::time_point last_query_time;
};

/**
 * @brief Query optimizer for similarity search operations
 * 
 * This service implements query cost calculation, index selection,
 * filter pushdown optimization, and query plan caching.
 */
class QueryOptimizer {
public:
    QueryOptimizer();
    ~QueryOptimizer() = default;
    
    /**
     * @brief Generate an optimal query plan
     * @param database_id Database to query
     * @param query_vector Query vector for cost estimation
     * @param params Search parameters
     * @return Optimal query plan
     */
    QueryOptimizationPlan generate_query_plan(const std::string& database_id,
                                 const Vector& query_vector,
                                 const SearchParams& params);
    
    /**
     * @brief Update index statistics
     * @param database_id Database identifier
     * @param index_type Type of index
     * @param stats Index statistics
     */
    void update_index_stats(const std::string& database_id,
                           IndexType index_type,
                           const IndexStats& stats);
    
    /**
     * @brief Record query execution statistics
     * @param database_id Database identifier
     * @param plan Query plan that was executed
     * @param actual_time_ms Actual execution time
     * @param vectors_scanned Number of vectors scanned
     * @param results_returned Number of results returned
     */
    void record_query_execution(const std::string& database_id,
                               const QueryOptimizationPlan& plan,
                               double actual_time_ms,
                               size_t vectors_scanned,
                               size_t results_returned);
    
    /**
     * @brief Get cached query plan if available
     * @param database_id Database identifier
     * @param params_hash Hash of search parameters
     * @return Cached query plan or nullptr
     */
    const QueryOptimizationPlan* get_cached_plan(const std::string& database_id,
                                    const std::string& params_hash) const;
    
    /**
     * @brief Calculate selectivity of filters
     * @param params Search parameters with filters
     * @param total_vectors Total number of vectors
     * @return Estimated selectivity (0.0 to 1.0)
     */
    double calculate_filter_selectivity(const SearchParams& params,
                                       size_t total_vectors) const;
    
    /**
     * @brief Determine optimal filter order for pushdown
     * @param params Search parameters with filters
     * @return Ordered list of filter names (most selective first)
     */
    std::vector<std::string> optimize_filter_order(const SearchParams& params) const;
    
    /**
     * @brief Clear cached query plans
     */
    void clear_plan_cache();
    
    /**
     * @brief Get statistics for a database
     * @param database_id Database identifier
     * @return Query statistics
     */
    QueryStats get_query_stats(const std::string& database_id) const;

private:
    // Index statistics per database
    std::unordered_map<std::string, std::unordered_map<IndexType, IndexStats>> index_stats_;
    
    // Query plan cache: database_id -> params_hash -> plan
    std::unordered_map<std::string, std::unordered_map<std::string, QueryOptimizationPlan>> plan_cache_;
    
    // Query statistics per database
    std::unordered_map<std::string, QueryStats> query_stats_;
    
    // Mutex for thread safety
    mutable std::mutex stats_mutex_;
    mutable std::mutex cache_mutex_;
    
    // Maximum cache size per database
    static constexpr size_t MAX_CACHE_SIZE = 1000;
    
    /**
     * @brief Calculate cost for using a specific index
     * @param index_type Type of index
     * @param stats Index statistics
     * @param params Search parameters
     * @return Estimated cost
     */
    double calculate_index_cost(IndexType index_type,
                               const IndexStats& stats,
                               const SearchParams& params) const;
    
    /**
     * @brief Select best index based on cost model
     * @param database_id Database identifier
     * @param params Search parameters
     * @return Selected index type
     */
    IndexType select_best_index(const std::string& database_id,
                               const SearchParams& params) const;
    
    /**
     * @brief Generate hash for search parameters
     * @param params Search parameters
     * @return Hash string
     */
    std::string hash_params(const SearchParams& params) const;
    
    /**
     * @brief Check if filter pushdown is beneficial
     * @param params Search parameters
     * @param selectivity Filter selectivity
     * @return True if pushdown should be used
     */
    bool should_use_filter_pushdown(const SearchParams& params,
                                   double selectivity) const;
    
    /**
     * @brief Evict oldest plans from cache if needed
     * @param database_id Database identifier
     */
    void evict_old_plans(const std::string& database_id);
};

} // namespace jadevectordb

#endif // JADEVECTORDB_QUERY_OPTIMIZER_H
