#include "load_balancer.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>

namespace jadevectordb {

LoadBalancer::LoadBalancer() {
    logger_ = logging::LoggerManager::get_logger("LoadBalancer");
}

LoadBalancer::~LoadBalancer() {
    stop();
}

bool LoadBalancer::initialize(const LoadBalancerConfig& config, std::shared_ptr<HealthMonitor> health_monitor) {
    try {
        config_ = config;
        health_monitor_ = health_monitor;
        LOG_INFO(logger_, "LoadBalancer initialized");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Initialization failed: " + std::string(e.what()));
        return false;
    }
}

Result<bool> LoadBalancer::start() {
    running_ = true;
    LOG_INFO(logger_, "LoadBalancer started");
    return true;
}

void LoadBalancer::stop() {
    running_ = false;
    LOG_INFO(logger_, "LoadBalancer stopped");
}

Result<bool> LoadBalancer::add_node(const std::string& node_id, const std::string& address) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    node_addresses_[node_id] = address;
    NodeStats stats;
    stats.node_id = node_id;
    stats.active_connections = 0;
    stats.weight = config_.node_weights.count(node_id) ? config_.node_weights[node_id] : 1.0;
    node_stats_[node_id] = stats;
    LOG_INFO(logger_, "Added node: " + node_id);
    return true;
}

Result<bool> LoadBalancer::remove_node(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    node_addresses_.erase(node_id);
    node_stats_.erase(node_id);
    LOG_INFO(logger_, "Removed node: " + node_id);
    return true;
}

Result<bool> LoadBalancer::update_node_weight(const std::string& node_id, double weight) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    auto it = node_stats_.find(node_id);
    if (it != node_stats_.end()) {
        it->second.weight = weight;
        return true;
    }
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Node not found");
}

Result<std::string> LoadBalancer::select_node() {
    try {
        auto healthy = get_healthy_nodes();
        if (healthy.empty()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "No healthy nodes available");
        }
        
        switch (config_.strategy) {
            case LoadBalancingStrategy::ROUND_ROBIN:
                return select_round_robin();
            case LoadBalancingStrategy::LEAST_CONNECTIONS:
                return select_least_connections();
            case LoadBalancingStrategy::LEAST_LOADED:
                return select_least_loaded();
            case LoadBalancingStrategy::WEIGHTED_ROUND_ROBIN:
                return select_weighted_round_robin();
            case LoadBalancingStrategy::RANDOM:
                return select_random();
            default:
                return select_round_robin();
        }
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to select node: " + std::string(e.what()));
    }
}

Result<std::string> LoadBalancer::select_node_for_shard(const std::string& shard_id) {
    // For consistent hashing
    std::hash<std::string> hasher;
    size_t hash = hasher(shard_id);
    
    auto healthy = get_healthy_nodes();
    if (healthy.empty()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "No healthy nodes");
    }
    
    return healthy[hash % healthy.size()];
}

void LoadBalancer::record_request(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    auto it = node_stats_.find(node_id);
    if (it != node_stats_.end()) {
        it->second.active_connections++;
        it->second.request_count++;
    }
}

void LoadBalancer::record_response(const std::string& node_id, int64_t latency_ms, bool error) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    auto it = node_stats_.find(node_id);
    if (it != node_stats_.end()) {
        it->second.active_connections--;
        it->second.response_time_ms = latency_ms;
        if (error) {
            it->second.error_count++;
        }
    }
}

std::vector<NodeStats> LoadBalancer::get_node_stats() {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    std::vector<NodeStats> stats;
    for (const auto& pair : node_stats_) {
        stats.push_back(pair.second);
    }
    return stats;
}

Result<NodeStats> LoadBalancer::get_node_stat(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    auto it = node_stats_.find(node_id);
    if (it != node_stats_.end()) {
        return it->second;
    }
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Node not found");
}

std::string LoadBalancer::select_round_robin() {
    auto healthy = get_healthy_nodes();
    if (healthy.empty()) return "";
    
    int index = round_robin_index_.fetch_add(1) % healthy.size();
    return healthy[index];
}

std::string LoadBalancer::select_least_connections() {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    auto healthy = get_healthy_nodes();
    if (healthy.empty()) return "";
    
    std::string selected = healthy[0];
    int min_connections = INT_MAX;
    
    for (const auto& node : healthy) {
        auto it = node_stats_.find(node);
        if (it != node_stats_.end() && it->second.active_connections < min_connections) {
            min_connections = it->second.active_connections;
            selected = node;
        }
    }
    
    return selected;
}

std::string LoadBalancer::select_least_loaded() {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    auto healthy = get_healthy_nodes();
    if (healthy.empty()) return "";
    
    std::string selected = healthy[0];
    double min_load = 100.0;
    
    for (const auto& node : healthy) {
        auto it = node_stats_.find(node);
        if (it != node_stats_.end()) {
            double load = (it->second.cpu_usage + it->second.memory_usage) / 2.0;
            if (load < min_load) {
                min_load = load;
                selected = node;
            }
        }
    }
    
    return selected;
}

std::string LoadBalancer::select_weighted_round_robin() {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    auto healthy = get_healthy_nodes();
    if (healthy.empty()) return "";
    
    // Simple weighted selection
    double total_weight = 0.0;
    for (const auto& node : healthy) {
        auto it = node_stats_.find(node);
        if (it != node_stats_.end()) {
            total_weight += it->second.weight;
        }
    }
    
    if (total_weight == 0.0) {
        return healthy[0];
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, total_weight);
    double random = dis(gen);
    
    double cumulative = 0.0;
    for (const auto& node : healthy) {
        auto it = node_stats_.find(node);
        if (it != node_stats_.end()) {
            cumulative += it->second.weight;
            if (random <= cumulative) {
                return node;
            }
        }
    }
    
    return healthy[0];
}

std::string LoadBalancer::select_random() {
    auto healthy = get_healthy_nodes();
    if (healthy.empty()) return "";
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, healthy.size() - 1);
    
    return healthy[dis(gen)];
}

std::vector<std::string> LoadBalancer::get_healthy_nodes() {
    std::vector<std::string> healthy;
    
    if (!health_monitor_) {
        // Return all nodes if no health monitor
        for (const auto& pair : node_addresses_) {
            healthy.push_back(pair.first);
        }
        return healthy;
    }
    
    for (const auto& pair : node_addresses_) {
        auto status_result = health_monitor_->get_node_status(pair.first);
        if (status_result.has_value() && status_result.value() == HealthStatus::HEALTHY) {
            healthy.push_back(pair.first);
        }
    }
    
    return healthy;
}

} // namespace jadevectordb
