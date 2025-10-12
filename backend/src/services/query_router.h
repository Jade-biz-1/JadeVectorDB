#ifndef JADEVECTORDB_QUERY_ROUTER_H
#define JADEVECTORDB_QUERY_ROUTER_H

#include "models/database.h"
#include "services/sharding_service.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace jadevectordb {

// Represents a query route
struct QueryRoute {
    std::string database_id;
    std::vector<std::string> target_shards;  // Shards to query
    std::vector<std::string> target_nodes;   // Nodes to send query to
    std::string query_type;                  // "search", "get", "store", etc.
    bool requires_aggregation;               // Whether results need to be aggregated
    
    QueryRoute() : requires_aggregation(false) {}
    QueryRoute(const std::string& db_id, const std::string& q_type)
        : database_id(db_id), query_type(q_type), requires_aggregation(false) {}
};

// Configuration for query routing
struct QueryRoutingConfig {
    bool enable_query_optimization;        // Whether to optimize query routing
    int max_parallel_queries;              // Maximum number of parallel queries
    int query_timeout_ms;                  // Timeout for individual queries
    std::string routing_algorithm;         // "direct", "adaptive", "load_balanced"
    bool enable_caching;                   // Whether to cache routing decisions
    
    QueryRoutingConfig() : 
        enable_query_optimization(true), 
        max_parallel_queries(10), 
        query_timeout_ms(5000),
        enable_caching(true) {}
};

/**
 * @brief Service for routing queries to appropriate shards and nodes
 * 
 * This service determines which nodes should handle specific queries
 * based on sharding strategy and cluster state.
 */
class QueryRouter {
private:
    std::shared_ptr<logging::Logger> logger_;
    std::shared_ptr<ShardingService> sharding_service_;
    QueryRoutingConfig config_;
    std::unordered_map<std::string, QueryRoute> route_cache_;  // query_pattern -> route
    mutable std::shared_mutex cache_mutex_;
    
public:
    explicit QueryRouter(std::shared_ptr<ShardingService> sharding_service);
    ~QueryRouter() = default;
    
    // Initialize the query router with configuration
    bool initialize(const QueryRoutingConfig& config);
    
    // Route a vector storage query
    Result<QueryRoute> route_store_query(const std::string& vector_id, 
                                       const std::string& database_id);
    
    // Route a vector retrieval query
    Result<QueryRoute> route_get_query(const std::string& vector_id, 
                                     const std::string& database_id);
    
    // Route a similarity search query
    Result<QueryRoute> route_search_query(const std::vector<float>& query_vector, 
                                        const std::string& database_id);
    
    // Route an advanced search query with filters
    Result<QueryRoute> route_advanced_search_query(const std::vector<float>& query_vector, 
                                                  const std::string& database_id,
                                                  const std::string& filters);
    
    // Route a database management query
    Result<QueryRoute> route_db_management_query(const std::string& database_id,
                                               const std::string& operation);
    
    // Route a batch operation query
    Result<QueryRoute> route_batch_query(const std::vector<std::string>& vector_ids,
                                       const std::string& database_id);
    
    // Get all nodes participating in a specific database
    Result<std::vector<std::string>> get_nodes_for_database(const std::string& database_id) const;
    
    // Update routing based on cluster changes
    Result<bool> update_routing_for_cluster_change();
    
    // Invalidate routing cache for a database
    void invalidate_cache_for_database(const std::string& database_id);
    
    // Get current routing configuration
    QueryRoutingConfig get_config() const;
    
    // Update routing configuration
    Result<bool> update_config(const QueryRoutingConfig& new_config);
    
    // Route a query based on complex criteria
    Result<QueryRoute> route_complex_query(const std::string& database_id,
                                         const std::string& operation_type,
                                         const std::unordered_map<std::string, std::string>& parameters);
    
    // Get statistics about query routing
    Result<std::unordered_map<std::string, int>> get_routing_stats() const;

private:
    // Determine target shards for an operation
    Result<std::vector<std::string>> determine_target_shards(const std::string& operation_type,
                                                           const std::string& database_id,
                                                           const std::string& key) const;
    
    // Apply routing algorithm to select nodes
    std::vector<std::string> apply_routing_algorithm(const std::vector<std::string>& candidate_nodes) const;
    
    // Check if route is cached
    bool is_route_cached(const std::string& query_pattern, QueryRoute& route) const;
    
    // Cache a route decision
    void cache_route(const std::string& query_pattern, const QueryRoute& route);
    
    // Validate query routing configuration
    bool validate_config(const QueryRoutingConfig& config) const;
    
    // Calculate load on nodes to make routing decisions
    std::unordered_map<std::string, double> calculate_node_loads() const;
    
    // Perform adaptive routing based on current cluster state
    Result<QueryRoute> adaptive_route(const std::string& operation_type,
                                    const std::string& database_id,
                                    const std::string& key) const;
    
    // Perform load-balanced routing
    std::vector<std::string> load_balanced_routing(const std::vector<std::string>& candidate_nodes,
                                                  int required_nodes) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_QUERY_ROUTER_H