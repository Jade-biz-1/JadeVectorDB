#pragma once

#include "lib/result.h"
#include "lib/logging.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <chrono>
#include <functional>
#include <atomic>

namespace jadevectordb {

// Health status enum
enum class HealthStatus {
    HEALTHY,
    DEGRADED,
    UNHEALTHY,
    UNKNOWN
};

// Node health information
struct NodeHealth {
    std::string node_id;
    HealthStatus status;
    double cpu_usage;
    double memory_usage;
    double disk_usage;
    int64_t last_heartbeat_ms;
    int64_t response_time_ms;
    bool is_reachable;
    std::string error_message;
    std::map<std::string, std::string> metadata;
};

// Cluster health information
struct ClusterHealth {
    HealthStatus overall_status;
    int total_nodes;
    int healthy_nodes;
    int degraded_nodes;
    int unhealthy_nodes;
    std::vector<NodeHealth> node_health;
    int64_t timestamp_ms;
    std::map<std::string, std::string> alerts;
};

// Health check configuration
struct HealthCheckConfig {
    int check_interval_seconds = 10;
    int timeout_seconds = 5;
    int unhealthy_threshold = 3;  // consecutive failures
    int degraded_threshold = 2;
    double cpu_threshold = 90.0;
    double memory_threshold = 90.0;
    double disk_threshold = 85.0;
    bool enable_auto_recovery = true;
    bool enable_alerts = true;
};

// Alert handler callback
using AlertHandler = std::function<void(const std::string& node_id, HealthStatus status, const std::string& message)>;

class HealthMonitor {
public:
    HealthMonitor();
    ~HealthMonitor();

    // Initialization and lifecycle
    bool initialize(const HealthCheckConfig& config);
    Result<bool> start();
    void stop();
    bool is_running() const { return running_; }

    // Node registration and management
    Result<bool> register_node(const std::string& node_id, const std::string& address);
    Result<bool> unregister_node(const std::string& node_id);
    std::vector<std::string> get_registered_nodes() const;

    // Health checks
    Result<NodeHealth> check_node_health(const std::string& node_id);
    Result<ClusterHealth> check_cluster_health();
    Result<HealthStatus> get_node_status(const std::string& node_id) const;
    
    // Update health information (called by nodes themselves)
    Result<bool> update_node_health(const std::string& node_id, const NodeHealth& health);
    Result<bool> record_heartbeat(const std::string& node_id);

    // Alert management
    void register_alert_handler(AlertHandler handler);
    std::vector<std::string> get_active_alerts() const;
    Result<bool> clear_alerts(const std::string& node_id);

    // Recovery actions
    Result<bool> trigger_recovery(const std::string& node_id);
    
private:
    // Monitoring thread
    void monitoring_loop();
    void check_all_nodes();
    void check_single_node(const std::string& node_id);
    void update_node_status(const std::string& node_id, HealthStatus new_status);
    
    // Alert handling
    void trigger_alert(const std::string& node_id, HealthStatus status, const std::string& message);
    void clear_node_alerts(const std::string& node_id);
    
    // Health evaluation
    HealthStatus evaluate_node_health(const NodeHealth& health) const;
    HealthStatus evaluate_cluster_health(const std::vector<NodeHealth>& nodes) const;
    
    // Helper functions
    int64_t get_current_timestamp_ms() const;
    bool is_node_responsive(const std::string& node_id) const;

    // Configuration
    HealthCheckConfig config_;
    mutable std::mutex config_mutex_;
    
    // Node tracking
    struct NodeInfo {
        std::string node_id;
        std::string address;
        NodeHealth health;
        int consecutive_failures;
        int64_t last_check_ms;
        std::vector<std::string> recent_errors;
    };
    std::map<std::string, NodeInfo> nodes_;
    mutable std::mutex nodes_mutex_;
    
    // Alert management
    std::map<std::string, std::vector<std::string>> active_alerts_;
    mutable std::mutex alerts_mutex_;
    std::vector<AlertHandler> alert_handlers_;
    mutable std::mutex handlers_mutex_;
    
    // Monitoring thread
    std::unique_ptr<std::thread> monitor_thread_;
    std::atomic<bool> running_{false};
    
    // Logger
    std::shared_ptr<logging::Logger> logger_;
};

// Helper functions to convert enums to strings
std::string health_status_to_string(HealthStatus status);
HealthStatus string_to_health_status(const std::string& status);

} // namespace jadevectordb
