#ifndef JADEVECTORDB_QUERY_ROUTER_H
#define JADEVECTORDB_QUERY_ROUTER_H

#include "models/database.h"
#include "models/vector.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>
#include <shared_mutex>

namespace jadevectordb {

// Represents a route in the distributed system
struct RouteInfo {
    std::string route_id;
    std::string database_id;
    std::string shard_id;
    std::string node_id;
    std::string operation_type;  // "read", "write", "search", etc.
    std::vector<std::string> target_nodes;  // Nodes that should handle this request
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point expires_at;
    std::string status;         // "active", "pending", "expired", etc.
    
    RouteInfo() : created_at(std::chrono::steady_clock::now()), status("active") {}
};

// Configuration for query routing
struct RoutingConfig {
    std::string strategy;          // "round_robin", "least_loaded", "consistent_hash", "adaptive"
    int max_route_cache_size;     // Maximum number of cached routes
    int route_ttl_seconds;        // Time-to-live for cached routes
    bool enable_adaptive_routing; // Whether to adjust routing based on node performance
    std::vector<std::string> preferred_nodes; // Preferred nodes for routing
    
    RoutingConfig() : max_route_cache_size(1000), route_ttl_seconds(300), 
                     enable_adaptive_routing(true) {}
};

// Search parameters for routing
struct RoutingSearchParams {
    int top_k;
    double threshold;
    bool include_metadata;
    bool include_vector_data;
    std::unordered_map<std::string, std::string> filters;
    
    RoutingSearchParams() : top_k(10), threshold(0.0), include_metadata(false), include_vector_data(false) {}
};

/**
 * @brief Service for routing queries to appropriate nodes in a distributed system
 * 
 * This service determines which nodes should handle specific database operations
 * based on sharding configuration, node health, and load balancing strategies.
 */
class QueryRouter {
public:
    enum class RoutingStrategy {
        ROUND_ROBIN,
        LEAST_LOADED,
        CONSISTENT_HASH,
        ADAPTIVE
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    RoutingConfig config_;
    mutable std::unordered_map<std::string, RouteInfo> route_cache_;  // route_key -> RouteInfo (mutable for cache in const methods)
    mutable std::unordered_map<std::string, int> node_load_;         // node_id -> current_load (mutable for metrics)
    mutable std::unordered_map<std::string, double> node_performance_; // node_id -> performance_score (mutable for metrics)
    mutable std::shared_mutex route_mutex_;
    mutable std::shared_mutex config_mutex_;
    
    // Round-robin counters for databases
    mutable std::unordered_map<std::string, size_t> round_robin_counters_; // database_id -> counter (mutable for routing state)
    
    // Consistent hash ring
    mutable std::vector<std::pair<uint64_t, std::string>> hash_ring_; // hash -> node_id (mutable for lazy init)
    mutable std::shared_mutex ring_mutex_;

public:
    explicit QueryRouter();
    ~QueryRouter() = default;
    
    // Initialize the query router with configuration
    bool initialize(const RoutingConfig& config);
    
    // Route a database operation to appropriate nodes
    Result<RouteInfo> route_operation(const std::string& database_id,
                                    const std::string& operation_type,
                                    const std::string& operation_key = "") const;
    
    // Route a vector operation to appropriate nodes
    Result<RouteInfo> route_vector_operation(const std::string& database_id,
                                          const std::string& vector_id,
                                          const std::string& operation_type) const;
    
    // Route a search operation to appropriate nodes
    Result<RouteInfo> route_search_operation(const std::string& database_id,
                                          const Vector& query_vector,
                                          const RoutingSearchParams& search_params) const;
    
    // Route a batch operation to appropriate nodes
    Result<RouteInfo> route_batch_operation(const std::string& database_id,
                                          const std::vector<std::string>& vector_ids,
                                          const std::string& operation_type) const;
    
    // Get cached route information for a specific operation
    Result<RouteInfo> get_cached_route(const std::string& route_key) const;
    
    // Update routing configuration
    Result<bool> update_routing_config(const RoutingConfig& new_config);
    
    // Update node load information
    Result<bool> update_node_load(const std::string& node_id, int load);
    
    // Update node performance metrics
    Result<bool> update_node_performance(const std::string& node_id, double performance_score);
    
    // Get current routing configuration
    RoutingConfig get_config() const;
    
    // Get statistics about routing decisions
    Result<std::unordered_map<std::string, size_t>> get_routing_stats() const;
    
    // Check if a route is still valid
    Result<bool> is_route_valid(const RouteInfo& route) const;
    
    // Invalidate a cached route
    Result<bool> invalidate_route(const std::string& route_key);
    
    // Clear expired routes from cache
    Result<bool> clear_expired_routes();
    
    // Get all candidate nodes for a database
    Result<std::vector<std::string>> get_candidate_nodes(const std::string& database_id) const;
    
    // Select the best node for an operation based on current strategy
    Result<std::string> select_best_node(const std::string& database_id,
                                       const std::string& operation_type,
                                       const std::string& operation_key = "") const;
    
    // Get routing strategy for a specific database
    RoutingStrategy get_strategy_for_database(const std::string& database_id) const;

private:
    // Generate a cache key for a route
    std::string generate_route_key(const std::string& database_id,
                                  const std::string& operation_type,
                                  const std::string& operation_key) const;
    
    // Route using round-robin strategy
    Result<RouteInfo> route_round_robin(const std::string& database_id,
                                      const std::string& operation_type,
                                      const std::string& operation_key) const;
    
    // Route using least-loaded strategy
    Result<RouteInfo> route_least_loaded(const std::string& database_id,
                                       const std::string& operation_type,
                                       const std::string& operation_key) const;
    
    // Route using consistent hash strategy
    Result<RouteInfo> route_consistent_hash(const std::string& database_id,
                                          const std::string& operation_type,
                                          const std::string& operation_key) const;
    
    // Route using adaptive strategy based on performance
    Result<RouteInfo> route_adaptive(const std::string& database_id,
                                   const std::string& operation_type,
                                   const std::string& operation_key) const;
    
    // Hash function for consistent hashing
    uint64_t hash_function(const std::string& key) const;
    
    // Build consistent hash ring
    void build_hash_ring(const std::vector<std::string>& nodes) const;
    
    // Get node from hash ring
    std::string get_node_from_ring(uint64_t hash) const;
    
    // Validate routing configuration
    bool validate_config(const RoutingConfig& config) const;
    
    // Select multiple nodes for replication or redundancy
    Result<std::vector<std::string>> select_multiple_nodes(const std::string& database_id,
                                                         const std::string& operation_type,
                                                         const std::string& operation_key,
                                                         int count) const;
    
    // Get preferred nodes for a database
    std::vector<std::string> get_preferred_nodes_for_database(const std::string& database_id) const;
    
    // Calculate load factor for a node
    double calculate_load_factor(const std::string& node_id) const;
    
    // Calculate performance factor for a node
    double calculate_performance_factor(const std::string& node_id) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_QUERY_ROUTER_H