#pragma once

#include "lib/result.h"
#include "lib/logging.h"
#include "health_monitor.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>

namespace jadevectordb {

// Load balancing strategies
enum class LoadBalancingStrategy {
    ROUND_ROBIN,
    LEAST_CONNECTIONS,
    LEAST_LOADED,
    WEIGHTED_ROUND_ROBIN,
    LOCALITY_AWARE,
    RANDOM
};

// Node stats for load balancing
struct NodeStats {
    std::string node_id;
    int active_connections;
    double cpu_usage;
    double memory_usage;
    int64_t response_time_ms;
    int request_count;
    int error_count;
    double weight;
};

// Load balancer configuration
struct LoadBalancerConfig {
    LoadBalancingStrategy strategy = LoadBalancingStrategy::ROUND_ROBIN;
    int max_connections_per_node = 1000;
    int health_check_interval_seconds = 10;
    int connection_timeout_seconds = 30;
    bool enable_sticky_sessions = false;
    std::map<std::string, double> node_weights;
};

class LoadBalancer {
public:
    LoadBalancer();
    ~LoadBalancer();
    
    bool initialize(const LoadBalancerConfig& config, std::shared_ptr<HealthMonitor> health_monitor);
    Result<bool> start();
    void stop();
    
    // Node management
    Result<bool> add_node(const std::string& node_id, const std::string& address);
    Result<bool> remove_node(const std::string& node_id);
    Result<bool> update_node_weight(const std::string& node_id, double weight);
    
    // Load balancing
    Result<std::string> select_node();
    Result<std::string> select_node_for_shard(const std::string& shard_id);
    void record_request(const std::string& node_id);
    void record_response(const std::string& node_id, int64_t latency_ms, bool error);
    
    // Stats
    std::vector<NodeStats> get_node_stats();
    Result<NodeStats> get_node_stat(const std::string& node_id);
    
private:
    std::string select_round_robin();
    std::string select_least_connections();
    std::string select_least_loaded();
    std::string select_weighted_round_robin();
    std::string select_random();
    std::vector<std::string> get_healthy_nodes();
    
    LoadBalancerConfig config_;
    std::shared_ptr<HealthMonitor> health_monitor_;
    
    std::map<std::string, NodeStats> node_stats_;
    std::map<std::string, std::string> node_addresses_;
    mutable std::mutex nodes_mutex_;
    
    std::atomic<int> round_robin_index_{0};
    std::atomic<bool> running_{false};
    
    std::shared_ptr<logging::Logger> logger_;
};

} // namespace jadevectordb
