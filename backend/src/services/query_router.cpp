#include "query_router.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <functional>

namespace jadevectordb {

QueryRouter::QueryRouter() {
    logger_ = logging::LoggerManager::get_logger("QueryRouter");
}

bool QueryRouter::initialize(const RoutingConfig& config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (!validate_config(config)) {
            LOG_ERROR(logger_, "Invalid routing configuration provided");
            return false;
        }
        
        config_ = config;
        
        LOG_INFO(logger_, "QueryRouter initialized with strategy: " + config_.strategy + 
                ", max_cache_size: " + std::to_string(config_.max_route_cache_size) + 
                ", ttl_seconds: " + std::to_string(config_.route_ttl_seconds));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in QueryRouter::initialize: " + std::string(e.what()));
        return false;
    }
}

Result<RouteInfo> QueryRouter::route_operation(const std::string& database_id,
                                             const std::string& operation_type,
                                             const std::string& operation_key) const {
    try {
        std::string route_key = generate_route_key(database_id, operation_type, operation_key);
        
        // Check cache first
        {
            std::lock_guard<std::mutex> lock(route_mutex_);
            auto it = route_cache_.find(route_key);
            if (it != route_cache_.end() && is_route_valid(it->second).value_or(true)) {
                LOG_DEBUG(logger_, "Using cached route for operation: " + route_key);
                return it->second;
            }
        }
        
        // Generate new route
        RouteInfo route_info;
        route_info.route_id = route_key;
        route_info.database_id = database_id;
        route_info.operation_type = operation_type;
        
        // Select appropriate nodes for this operation
        auto nodes_result = select_multiple_nodes(database_id, operation_type, operation_key, 1);
        if (!nodes_result.has_value()) {
            LOG_ERROR(logger_, "Failed to select nodes for operation: " + route_key);
            return nodes_result;
        }
        
        route_info.target_nodes = nodes_result.value();
        route_info.created_at = std::chrono::steady_clock::now();
        route_info.expires_at = route_info.created_at + std::chrono::seconds(config_.route_ttl_seconds);
        
        // Cache the route
        {
            std::lock_guard<std::mutex> lock(route_mutex_);
            // Evict oldest entries if cache is full
            if (route_cache_.size() >= static_cast<size_t>(config_.max_route_cache_size)) {
                auto oldest_it = route_cache_.begin();
                auto oldest_time = oldest_it->second.created_at;
                for (auto it = route_cache_.begin(); it != route_cache_.end(); ++it) {
                    if (it->second.created_at < oldest_time) {
                        oldest_it = it;
                        oldest_time = it->second.created_at;
                    }
                }
                route_cache_.erase(oldest_it);
            }
            route_cache_[route_key] = route_info;
        }
        
        LOG_DEBUG(logger_, "Generated new route for operation: " + route_key + 
                 " to node: " + (route_info.target_nodes.empty() ? "none" : route_info.target_nodes[0]));
        return route_info;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in route_operation: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to route operation: " + std::string(e.what()));
    }
}

Result<RouteInfo> QueryRouter::route_vector_operation(const std::string& database_id,
                                                   const std::string& vector_id,
                                                   const std::string& operation_type) const {
    try {
        LOG_DEBUG(logger_, "Routing vector operation " + operation_type + " for vector " + vector_id + 
                 " in database " + database_id);
        
        return route_operation(database_id, operation_type, vector_id);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in route_vector_operation: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to route vector operation: " + std::string(e.what()));
    }
}

Result<RouteInfo> QueryRouter::route_search_operation(const std::string& database_id,
                                                   const Vector& query_vector,
                                                   const SearchParams& search_params) const {
    try {
        LOG_DEBUG(logger_, "Routing search operation for database " + database_id);
        
        // For search operations, we might need to route to multiple nodes depending on sharding
        // For now, we'll route to a single node based on the first vector in the query
        std::string operation_key = query_vector.id.empty() ? "search_query" : query_vector.id;
        return route_operation(database_id, "search", operation_key);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in route_search_operation: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to route search operation: " + std::string(e.what()));
    }
}

Result<RouteInfo> QueryRouter::route_batch_operation(const std::string& database_id,
                                                   const std::vector<std::string>& vector_ids,
                                                   const std::string& operation_type) const {
    try {
        LOG_DEBUG(logger_, "Routing batch operation " + operation_type + " for " + 
                 std::to_string(vector_ids.size()) + " vectors in database " + database_id);
        
        // For batch operations, we might need to route to multiple nodes
        // For now, we'll route to a single node based on the first vector in the batch
        std::string operation_key = vector_ids.empty() ? "batch_operation" : vector_ids[0];
        return route_operation(database_id, operation_type, operation_key);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in route_batch_operation: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to route batch operation: " + std::string(e.what()));
    }
}

Result<RouteInfo> QueryRouter::get_cached_route(const std::string& route_key) const {
    try {
        std::lock_guard<std::mutex> lock(route_mutex_);
        auto it = route_cache_.find(route_key);
        if (it != route_cache_.end()) {
            if (is_route_valid(it->second).value_or(true)) {
                return it->second;
            } else {
                // Expired route, remove it
                route_cache_.erase(it);
                LOG_DEBUG(logger_, "Removed expired cached route: " + route_key);
                RETURN_ERROR(ErrorCode::RESOURCE_EXPIRED, "Route has expired: " + route_key);
            }
        }
        
        LOG_DEBUG(logger_, "Route not found in cache: " + route_key);
        RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Route not found: " + route_key);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_cached_route: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get cached route: " + std::string(e.what()));
    }
}

Result<bool> QueryRouter::update_routing_config(const RoutingConfig& new_config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (!validate_config(new_config)) {
            LOG_ERROR(logger_, "Invalid routing configuration provided for update");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid routing configuration");
        }
        
        config_ = new_config;
        
        LOG_INFO(logger_, "Updated routing configuration: strategy=" + config_.strategy + 
                ", max_cache_size=" + std::to_string(config_.max_route_cache_size) + 
                ", ttl_seconds=" + std::to_string(config_.route_ttl_seconds));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_routing_config: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update routing configuration: " + std::string(e.what()));
    }
}

Result<bool> QueryRouter::update_node_load(const std::string& node_id, int load) {
    try {
        std::lock_guard<std::mutex> lock(route_mutex_);
        node_load_[node_id] = load;
        
        LOG_DEBUG(logger_, "Updated load for node " + node_id + " to " + std::to_string(load));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_node_load: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update node load: " + std::string(e.what()));
    }
}

Result<bool> QueryRouter::update_node_performance(const std::string& node_id, double performance_score) {
    try {
        std::lock_guard<std::mutex> lock(route_mutex_);
        node_performance_[node_id] = performance_score;
        
        LOG_DEBUG(logger_, "Updated performance score for node " + node_id + " to " + std::to_string(performance_score));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_node_performance: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update node performance: " + std::string(e.what()));
    }
}

RoutingConfig QueryRouter::get_config() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

Result<std::unordered_map<std::string, size_t>> QueryRouter::get_routing_stats() const {
    try {
        std::lock_guard<std::mutex> lock(route_mutex_);
        
        std::unordered_map<std::string, size_t> stats;
        stats["cached_routes"] = route_cache_.size();
        stats["node_load_entries"] = node_load_.size();
        stats["node_performance_entries"] = node_performance_.size();
        
        // Count routes by operation type
        std::unordered_map<std::string, size_t> operation_counts;
        for (const auto& entry : route_cache_) {
            operation_counts[entry.second.operation_type]++;
        }
        stats["read_operations"] = operation_counts["read"];
        stats["write_operations"] = operation_counts["write"];
        stats["search_operations"] = operation_counts["search"];
        stats["batch_operations"] = operation_counts["batch"];
        
        LOG_DEBUG(logger_, "Generated routing statistics");
        return stats;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_routing_stats: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get routing statistics: " + std::string(e.what()));
    }
}

Result<bool> QueryRouter::is_route_valid(const RouteInfo& route) const {
    try {
        auto now = std::chrono::steady_clock::now();
        bool is_valid = route.expires_at > now;
        
        if (!is_valid) {
            LOG_DEBUG(logger_, "Route " + route.route_id + " has expired");
        }
        
        return is_valid;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in is_route_valid: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to validate route: " + std::string(e.what()));
    }
}

Result<bool> QueryRouter::invalidate_route(const std::string& route_key) {
    try {
        std::lock_guard<std::mutex> lock(route_mutex_);
        auto erased_count = route_cache_.erase(route_key);
        
        bool invalidated = erased_count > 0;
        if (invalidated) {
            LOG_DEBUG(logger_, "Invalidated cached route: " + route_key);
        } else {
            LOG_DEBUG(logger_, "Route not found for invalidation: " + route_key);
        }
        
        return invalidated;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in invalidate_route: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to invalidate route: " + std::string(e.what()));
    }
}

Result<bool> QueryRouter::clear_expired_routes() {
    try {
        std::lock_guard<std::mutex> lock(route_mutex_);
        size_t initial_size = route_cache_.size();
        size_t removed_count = 0;
        
        auto now = std::chrono::steady_clock::now();
        for (auto it = route_cache_.begin(); it != route_cache_.end();) {
            if (it->second.expires_at <= now) {
                it = route_cache_.erase(it);
                removed_count++;
            } else {
                ++it;
            }
        }
        
        LOG_INFO(logger_, "Cleared " + std::to_string(removed_count) + " expired routes from cache");
        return removed_count > 0;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in clear_expired_routes: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to clear expired routes: " + std::string(e.what()));
    }
}

Result<std::vector<std::string>> QueryRouter::get_candidate_nodes(const std::string& database_id) const {
    try {
        // In a real implementation, this would query the cluster service
        // For now, we'll return a list of mock nodes
        std::vector<std::string> nodes;
        
        if (!config_.preferred_nodes.empty()) {
            nodes = config_.preferred_nodes;
        } else {
            // Generate mock nodes
            for (int i = 0; i < 5; ++i) {
                nodes.push_back("node_" + database_id + "_" + std::to_string(i));
            }
        }
        
        LOG_DEBUG(logger_, "Found " + std::to_string(nodes.size()) + " candidate nodes for database " + database_id);
        return nodes;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_candidate_nodes: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get candidate nodes: " + std::string(e.what()));
    }
}

Result<std::string> QueryRouter::select_best_node(const std::string& database_id,
                                                const std::string& operation_type,
                                                const std::string& operation_key) const {
    try {
        auto nodes_result = get_candidate_nodes(database_id);
        if (!nodes_result.has_value()) {
            LOG_ERROR(logger_, "Failed to get candidate nodes for database: " + database_id);
            return nodes_result;
        }
        
        auto nodes = nodes_result.value();
        if (nodes.empty()) {
            LOG_WARN(logger_, "No candidate nodes found for database: " + database_id);
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "No candidate nodes available for database: " + database_id);
        }
        
        // Select node based on routing strategy
        if (config_.strategy == "round_robin") {
            return route_round_robin(database_id, operation_type, operation_key).value().target_nodes[0];
        } else if (config_.strategy == "least_loaded") {
            return route_least_loaded(database_id, operation_type, operation_key).value().target_nodes[0];
        } else if (config_.strategy == "consistent_hash") {
            return route_consistent_hash(database_id, operation_type, operation_key).value().target_nodes[0];
        } else if (config_.strategy == "adaptive") {
            return route_adaptive(database_id, operation_type, operation_key).value().target_nodes[0];
        } else {
            // Default to round-robin
            return route_round_robin(database_id, operation_type, operation_key).value().target_nodes[0];
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in select_best_node: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to select best node: " + std::string(e.what()));
    }
}

QueryRouter::RoutingStrategy QueryRouter::get_strategy_for_database(const std::string& database_id) const {
    if (config_.strategy == "round_robin") {
        return RoutingStrategy::ROUND_ROBIN;
    } else if (config_.strategy == "least_loaded") {
        return RoutingStrategy::LEAST_LOADED;
    } else if (config_.strategy == "consistent_hash") {
        return RoutingStrategy::CONSISTENT_HASH;
    } else if (config_.strategy == "adaptive") {
        return RoutingStrategy::ADAPTIVE;
    } else {
        return RoutingStrategy::ROUND_ROBIN; // Default
    }
}

// Private methods

std::string QueryRouter::generate_route_key(const std::string& database_id,
                                          const std::string& operation_type,
                                          const std::string& operation_key) const {
    return database_id + ":" + operation_type + ":" + operation_key;
}

Result<RouteInfo> QueryRouter::route_round_robin(const std::string& database_id,
                                               const std::string& operation_type,
                                               const std::string& operation_key) const {
    try {
        auto nodes_result = get_candidate_nodes(database_id);
        if (!nodes_result.has_value()) {
            return nodes_result;
        }
        
        auto nodes = nodes_result.value();
        if (nodes.empty()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "No nodes available for round-robin routing");
        }
        
        RouteInfo route_info;
        
        // Get the round-robin counter for this database
        std::lock_guard<std::mutex> lock(route_mutex_);
        size_t& counter = round_robin_counters_[database_id];
        route_info.target_nodes.push_back(nodes[counter % nodes.size()]);
        counter++; // Increment for next time
        
        return route_info;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in route_round_robin: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to route using round-robin: " + std::string(e.what()));
    }
}

Result<RouteInfo> QueryRouter::route_least_loaded(const std::string& database_id,
                                                const std::string& operation_type,
                                                const std::string& operation_key) const {
    try {
        auto nodes_result = get_candidate_nodes(database_id);
        if (!nodes_result.has_value()) {
            return nodes_result;
        }
        
        auto nodes = nodes_result.value();
        if (nodes.empty()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "No nodes available for least-loaded routing");
        }
        
        RouteInfo route_info;
        
        // Find the node with the least load
        std::lock_guard<std::mutex> lock(route_mutex_);
        std::string best_node = nodes[0];
        int min_load = std::numeric_limits<int>::max();
        
        for (const auto& node : nodes) {
            int load = node_load_[node];
            if (load < min_load) {
                min_load = load;
                best_node = node;
            }
        }
        
        route_info.target_nodes.push_back(best_node);
        return route_info;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in route_least_loaded: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to route using least-loaded: " + std::string(e.what()));
    }
}

Result<RouteInfo> QueryRouter::route_consistent_hash(const std::string& database_id,
                                                   const std::string& operation_type,
                                                   const std::string& operation_key) const {
    try {
        auto nodes_result = get_candidate_nodes(database_id);
        if (!nodes_result.has_value()) {
            return nodes_result;
        }
        
        auto nodes = nodes_result.value();
        if (nodes.empty()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "No nodes available for consistent hash routing");
        }
        
        RouteInfo route_info;
        
        // Build hash ring if it doesn't exist or is empty
        {
            std::lock_guard<std::mutex> lock(ring_mutex_);
            if (hash_ring_.empty() || hash_ring_.size() != nodes.size()) {
                build_hash_ring(nodes);
            }
        }
        
        // Hash the operation key and find the appropriate node
        uint64_t hash = hash_function(operation_key.empty() ? database_id : operation_key);
        std::string node = get_node_from_ring(hash);
        route_info.target_nodes.push_back(node);
        
        return route_info;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in route_consistent_hash: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to route using consistent hash: " + std::string(e.what()));
    }
}

Result<RouteInfo> QueryRouter::route_adaptive(const std::string& database_id,
                                           const std::string& operation_type,
                                           const std::string& operation_key) const {
    try {
        auto nodes_result = get_candidate_nodes(database_id);
        if (!nodes_result.has_value()) {
            return nodes_result;
        }
        
        auto nodes = nodes_result.value();
        if (nodes.empty()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "No nodes available for adaptive routing");
        }
        
        RouteInfo route_info;
        
        // Select node based on a combination of load and performance
        std::lock_guard<std::mutex> lock(route_mutex_);
        std::string best_node = nodes[0];
        double best_score = -1.0;
        
        for (const auto& node : nodes) {
            double load_factor = calculate_load_factor(node);
            double perf_factor = calculate_performance_factor(node);
            
            // Combine load and performance factors (lower load and higher performance are better)
            double score = (1.0 - load_factor) * perf_factor;
            if (score > best_score) {
                best_score = score;
                best_node = node;
            }
        }
        
        route_info.target_nodes.push_back(best_node);
        return route_info;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in route_adaptive: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to route using adaptive strategy: " + std::string(e.what()));
    }
}

uint64_t QueryRouter::hash_function(const std::string& key) const {
    // Simple hash function for consistent hashing
    uint64_t hash = 5381;
    for (char c : key) {
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    }
    return hash;
}

void QueryRouter::build_hash_ring(const std::vector<std::string>& nodes) {
    hash_ring_.clear();
    
    // Add each node multiple times to the ring for better distribution
    for (const auto& node : nodes) {
        for (int i = 0; i < 160; ++i) { // 160 virtual nodes per physical node
            std::string vnode_key = node + "#" + std::to_string(i);
            uint64_t hash = hash_function(vnode_key);
            hash_ring_.push_back({hash, node});
        }
    }
    
    // Sort the ring by hash value
    std::sort(hash_ring_.begin(), hash_ring_.end());
    
    LOG_DEBUG(logger_, "Built consistent hash ring with " + std::to_string(hash_ring_.size()) + " virtual nodes");
}

std::string QueryRouter::get_node_from_ring(uint64_t hash) const {
    if (hash_ring_.empty()) {
        return "default_node";
    }
    
    // Find the first node with a hash greater than or equal to the given hash
    auto it = std::lower_bound(hash_ring_.begin(), hash_ring_.end(), std::make_pair(hash, std::string("")));
    if (it == hash_ring_.end()) {
        it = hash_ring_.begin(); // Wrap around to the beginning
    }
    
    return it->second;
}

bool QueryRouter::validate_config(const RoutingConfig& config) const {
    // Basic validation
    if (config.max_route_cache_size < 0) {
        LOG_ERROR(logger_, "Invalid max_route_cache_size: " + std::to_string(config.max_route_cache_size));
        return false;
    }
    
    if (config.route_ttl_seconds < 0) {
        LOG_ERROR(logger_, "Invalid route_ttl_seconds: " + std::to_string(config.route_ttl_seconds));
        return false;
    }
    
    // Validate strategy
    if (!config.strategy.empty() && 
        config.strategy != "round_robin" && 
        config.strategy != "least_loaded" && 
        config.strategy != "consistent_hash" && 
        config.strategy != "adaptive") {
        LOG_ERROR(logger_, "Invalid routing strategy: " + config.strategy);
        return false;
    }
    
    return true;
}

Result<std::vector<std::string>> QueryRouter::select_multiple_nodes(const std::string& database_id,
                                                                 const std::string& operation_type,
                                                                 const std::string& operation_key,
                                                                 int count) const {
    try {
        auto nodes_result = get_candidate_nodes(database_id);
        if (!nodes_result.has_value()) {
            return nodes_result;
        }
        
        auto nodes = nodes_result.value();
        if (nodes.empty()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "No nodes available for selection");
        }
        
        // Just return the first 'count' nodes for now
        // In a real implementation, this would be more sophisticated
        std::vector<std::string> selected_nodes;
        for (int i = 0; i < std::min(count, static_cast<int>(nodes.size())); ++i) {
            selected_nodes.push_back(nodes[i]);
        }
        
        return selected_nodes;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in select_multiple_nodes: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to select multiple nodes: " + std::string(e.what()));
    }
}

std::vector<std::string> QueryRouter::get_preferred_nodes_for_database(const std::string& database_id) const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_.preferred_nodes;
}

double QueryRouter::calculate_load_factor(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(route_mutex_);
    auto it = node_load_.find(node_id);
    if (it != node_load_.end()) {
        // Normalize load to a factor between 0 and 1
        // Assuming maximum load of 1000 for normalization
        return std::min(1.0, static_cast<double>(it->second) / 1000.0);
    }
    return 0.0; // No load data, assume zero load
}

double QueryRouter::calculate_performance_factor(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(route_mutex_);
    auto it = node_performance_.find(node_id);
    if (it != node_performance_.end()) {
        // Performance score is already normalized between 0 and 1
        return it->second;
    }
    return 1.0; // No performance data, assume perfect performance
}

} // namespace jadevectordb