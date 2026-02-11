#include "health_monitor.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <sstream>
#include <cstring>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

namespace jadevectordb {

std::string health_status_to_string(HealthStatus status) {
    switch (status) {
        case HealthStatus::HEALTHY: return "HEALTHY";
        case HealthStatus::DEGRADED: return "DEGRADED";
        case HealthStatus::UNHEALTHY: return "UNHEALTHY";
        case HealthStatus::UNKNOWN: return "UNKNOWN";
        default: return "UNKNOWN";
    }
}

HealthStatus string_to_health_status(const std::string& status) {
    if (status == "HEALTHY") return HealthStatus::HEALTHY;
    if (status == "DEGRADED") return HealthStatus::DEGRADED;
    if (status == "UNHEALTHY") return HealthStatus::UNHEALTHY;
    return HealthStatus::UNKNOWN;
}

HealthMonitor::HealthMonitor() {
    logger_ = logging::LoggerManager::get_logger("HealthMonitor");
}

HealthMonitor::~HealthMonitor() {
    stop();
}

bool HealthMonitor::initialize(const HealthCheckConfig& config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (config.check_interval_seconds <= 0) {
            LOG_ERROR(logger_, "Invalid check interval: " + std::to_string(config.check_interval_seconds));
            return false;
        }
        
        if (config.timeout_seconds <= 0) {
            LOG_ERROR(logger_, "Invalid timeout: " + std::to_string(config.timeout_seconds));
            return false;
        }
        
        config_ = config;
        
        LOG_INFO(logger_, "HealthMonitor initialized - interval: " + 
                std::to_string(config_.check_interval_seconds) + "s, " +
                "timeout: " + std::to_string(config_.timeout_seconds) + "s");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in initialize: " + std::string(e.what()));
        return false;
    }
}

Result<bool> HealthMonitor::start() {
    try {
        if (running_) {
            LOG_WARN(logger_, "HealthMonitor already running");
            return true;
        }
        
        LOG_INFO(logger_, "Starting HealthMonitor");
        
        running_ = true;
        monitor_thread_ = std::make_unique<std::thread>(&HealthMonitor::monitoring_loop, this);
        
        LOG_INFO(logger_, "HealthMonitor started successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in start: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to start health monitor: " + std::string(e.what()));
    }
}

void HealthMonitor::stop() {
    try {
        if (!running_) {
            return;
        }
        
        LOG_INFO(logger_, "Stopping HealthMonitor");
        
        running_ = false;
        
        if (monitor_thread_ && monitor_thread_->joinable()) {
            monitor_thread_->join();
        }
        
        monitor_thread_.reset();
        
        LOG_INFO(logger_, "HealthMonitor stopped");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in stop: " + std::string(e.what()));
    }
}

Result<bool> HealthMonitor::register_node(const std::string& node_id, const std::string& address) {
    try {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        
        if (node_id.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Node ID cannot be empty");
        }
        
        if (nodes_.find(node_id) != nodes_.end()) {
            LOG_WARN(logger_, "Node already registered: " + node_id);
            return true;
        }
        
        NodeInfo info;
        info.node_id = node_id;
        info.address = address;
        info.health.node_id = node_id;
        info.health.status = HealthStatus::UNKNOWN;
        info.health.is_reachable = false;
        info.consecutive_failures = 0;
        info.last_check_ms = 0;
        
        nodes_[node_id] = info;
        
        LOG_INFO(logger_, "Registered node: " + node_id + " at " + address);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in register_node: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to register node: " + std::string(e.what()));
    }
}

Result<bool> HealthMonitor::unregister_node(const std::string& node_id) {
    try {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        
        auto it = nodes_.find(node_id);
        if (it == nodes_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Node not found: " + node_id);
        }
        
        nodes_.erase(it);
        clear_node_alerts(node_id);
        
        LOG_INFO(logger_, "Unregistered node: " + node_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in unregister_node: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to unregister node: " + std::string(e.what()));
    }
}

std::vector<std::string> HealthMonitor::get_registered_nodes() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    std::vector<std::string> node_ids;
    node_ids.reserve(nodes_.size());
    for (const auto& pair : nodes_) {
        node_ids.push_back(pair.first);
    }
    return node_ids;
}

Result<NodeHealth> HealthMonitor::check_node_health(const std::string& node_id) {
    try {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        
        auto it = nodes_.find(node_id);
        if (it == nodes_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Node not found: " + node_id);
        }
        
        return it->second.health;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in check_node_health: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check node health: " + std::string(e.what()));
    }
}

Result<ClusterHealth> HealthMonitor::check_cluster_health() {
    try {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        
        ClusterHealth cluster;
        cluster.total_nodes = nodes_.size();
        cluster.healthy_nodes = 0;
        cluster.degraded_nodes = 0;
        cluster.unhealthy_nodes = 0;
        cluster.timestamp_ms = get_current_timestamp_ms();
        
        for (const auto& pair : nodes_) {
            const auto& node_info = pair.second;
            cluster.node_health.push_back(node_info.health);
            
            switch (node_info.health.status) {
                case HealthStatus::HEALTHY:
                    cluster.healthy_nodes++;
                    break;
                case HealthStatus::DEGRADED:
                    cluster.degraded_nodes++;
                    break;
                case HealthStatus::UNHEALTHY:
                case HealthStatus::UNKNOWN:
                    cluster.unhealthy_nodes++;
                    break;
            }
        }
        
        cluster.overall_status = evaluate_cluster_health(cluster.node_health);
        
        // Add active alerts
        std::lock_guard<std::mutex> alert_lock(alerts_mutex_);
        for (const auto& pair : active_alerts_) {
            for (const auto& alert : pair.second) {
                cluster.alerts[pair.first] = alert;
            }
        }
        
        return cluster;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in check_cluster_health: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check cluster health: " + std::string(e.what()));
    }
}

Result<HealthStatus> HealthMonitor::get_node_status(const std::string& node_id) const {
    try {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        
        auto it = nodes_.find(node_id);
        if (it == nodes_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Node not found: " + node_id);
        }
        
        return it->second.health.status;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_node_status: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get node status: " + std::string(e.what()));
    }
}

Result<bool> HealthMonitor::update_node_health(const std::string& node_id, const NodeHealth& health) {
    try {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        
        auto it = nodes_.find(node_id);
        if (it == nodes_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Node not found: " + node_id);
        }
        
        NodeInfo& info = it->second;
        HealthStatus old_status = info.health.status;
        info.health = health;
        info.health.node_id = node_id;
        info.last_check_ms = get_current_timestamp_ms();
        
        // Evaluate the new health
        HealthStatus new_status = evaluate_node_health(health);
        info.health.status = new_status;
        
        // Update consecutive failures
        if (new_status == HealthStatus::UNHEALTHY) {
            info.consecutive_failures++;
        } else {
            info.consecutive_failures = 0;
        }
        
        // Trigger alerts if status changed
        if (old_status != new_status) {
            update_node_status(node_id, new_status);
        }
        
        LOG_DEBUG(logger_, "Updated health for node " + node_id + ": " + health_status_to_string(new_status));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_node_health: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update node health: " + std::string(e.what()));
    }
}

Result<bool> HealthMonitor::record_heartbeat(const std::string& node_id) {
    try {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        
        auto it = nodes_.find(node_id);
        if (it == nodes_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Node not found: " + node_id);
        }
        
        NodeInfo& info = it->second;
        info.health.last_heartbeat_ms = get_current_timestamp_ms();
        info.health.is_reachable = true;
        info.consecutive_failures = 0;
        
        // If node was unhealthy, mark as healthy
        if (info.health.status == HealthStatus::UNHEALTHY || 
            info.health.status == HealthStatus::UNKNOWN) {
            HealthStatus old_status = info.health.status;
            info.health.status = HealthStatus::HEALTHY;
            update_node_status(node_id, HealthStatus::HEALTHY);
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in record_heartbeat: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to record heartbeat: " + std::string(e.what()));
    }
}

void HealthMonitor::register_alert_handler(AlertHandler handler) {
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    alert_handlers_.push_back(handler);
    LOG_INFO(logger_, "Registered alert handler");
}

std::vector<std::string> HealthMonitor::get_active_alerts() const {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    std::vector<std::string> alerts;
    for (const auto& pair : active_alerts_) {
        for (const auto& alert : pair.second) {
            alerts.push_back(pair.first + ": " + alert);
        }
    }
    return alerts;
}

Result<bool> HealthMonitor::clear_alerts(const std::string& node_id) {
    try {
        clear_node_alerts(node_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in clear_alerts: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to clear alerts: " + std::string(e.what()));
    }
}

Result<bool> HealthMonitor::trigger_recovery(const std::string& node_id) {
    try {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        
        auto it = nodes_.find(node_id);
        if (it == nodes_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Node not found: " + node_id);
        }
        
        LOG_INFO(logger_, "Triggering recovery for node: " + node_id);
        
        // Reset failure count
        it->second.consecutive_failures = 0;
        
        // Trigger recovery alert
        trigger_alert(node_id, HealthStatus::DEGRADED, "Recovery triggered for node");
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in trigger_recovery: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to trigger recovery: " + std::string(e.what()));
    }
}

void HealthMonitor::monitoring_loop() {
    LOG_INFO(logger_, "Health monitoring loop started");
    
    while (running_) {
        try {
            check_all_nodes();
            
            // Sleep for the configured interval
            std::this_thread::sleep_for(std::chrono::seconds(config_.check_interval_seconds));
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Exception in monitoring_loop: " + std::string(e.what()));
        }
    }
    
    LOG_INFO(logger_, "Health monitoring loop stopped");
}

void HealthMonitor::check_all_nodes() {
    std::vector<std::string> node_ids;
    {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        for (const auto& pair : nodes_) {
            node_ids.push_back(pair.first);
        }
    }
    
    for (const auto& node_id : node_ids) {
        check_single_node(node_id);
    }
}

void HealthMonitor::check_single_node(const std::string& node_id) {
    try {
        std::lock_guard<std::mutex> lock(nodes_mutex_);
        
        auto it = nodes_.find(node_id);
        if (it == nodes_.end()) {
            return;
        }
        
        NodeInfo& info = it->second;
        int64_t now = get_current_timestamp_ms();
        int64_t heartbeat_age = now - info.health.last_heartbeat_ms;
        
        // Check if heartbeat is too old
        int64_t max_heartbeat_age = config_.timeout_seconds * 1000 * 2; // 2x timeout
        if (heartbeat_age > max_heartbeat_age) {
            info.health.is_reachable = false;
            info.consecutive_failures++;
            
            if (info.consecutive_failures >= config_.unhealthy_threshold) {
                HealthStatus old_status = info.health.status;
                info.health.status = HealthStatus::UNHEALTHY;
                info.health.error_message = "No heartbeat for " + std::to_string(heartbeat_age / 1000) + " seconds";
                
                if (old_status != HealthStatus::UNHEALTHY) {
                    update_node_status(node_id, HealthStatus::UNHEALTHY);
                }
            } else if (info.consecutive_failures >= config_.degraded_threshold) {
                HealthStatus old_status = info.health.status;
                info.health.status = HealthStatus::DEGRADED;
                info.health.error_message = "Degraded heartbeat";
                
                if (old_status != HealthStatus::DEGRADED && old_status != HealthStatus::UNHEALTHY) {
                    update_node_status(node_id, HealthStatus::DEGRADED);
                }
            }
        } else {
            // Node is responsive
            info.health.is_reachable = true;
            info.consecutive_failures = 0;
            
            // Check resource thresholds
            HealthStatus resource_status = evaluate_node_health(info.health);
            if (info.health.status != resource_status) {
                info.health.status = resource_status;
                update_node_status(node_id, resource_status);
            }
        }
        
        info.last_check_ms = now;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception checking node " + node_id + ": " + std::string(e.what()));
    }
}

void HealthMonitor::update_node_status(const std::string& node_id, HealthStatus new_status) {
    std::string status_str = health_status_to_string(new_status);
    LOG_INFO(logger_, "Node " + node_id + " status changed to: " + status_str);
    
    // Trigger alerts based on status
    if (new_status == HealthStatus::UNHEALTHY) {
        trigger_alert(node_id, new_status, "Node is unhealthy");
        
        // Trigger auto-recovery if enabled
        if (config_.enable_auto_recovery) {
            LOG_INFO(logger_, "Auto-recovery enabled for node: " + node_id);
            // Recovery logic would be triggered here
        }
    } else if (new_status == HealthStatus::DEGRADED) {
        trigger_alert(node_id, new_status, "Node is degraded");
    } else if (new_status == HealthStatus::HEALTHY) {
        // Clear alerts when node becomes healthy
        clear_node_alerts(node_id);
    }
}

void HealthMonitor::trigger_alert(const std::string& node_id, HealthStatus status, const std::string& message) {
    if (!config_.enable_alerts) {
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(alerts_mutex_);
        active_alerts_[node_id].push_back(message);
        
        // Keep only last 10 alerts per node
        if (active_alerts_[node_id].size() > 10) {
            active_alerts_[node_id].erase(active_alerts_[node_id].begin());
        }
    }
    
    LOG_WARN(logger_, "ALERT: Node " + node_id + " - " + message);
    
    // Call registered alert handlers
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    for (const auto& handler : alert_handlers_) {
        try {
            handler(node_id, status, message);
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Exception in alert handler: " + std::string(e.what()));
        }
    }
}

void HealthMonitor::clear_node_alerts(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    active_alerts_.erase(node_id);
    LOG_INFO(logger_, "Cleared alerts for node: " + node_id);
}

HealthStatus HealthMonitor::evaluate_node_health(const NodeHealth& health) const {
    // Check if node is unreachable
    if (!health.is_reachable) {
        return HealthStatus::UNHEALTHY;
    }
    
    // Check resource thresholds
    bool cpu_critical = health.cpu_usage > config_.cpu_threshold;
    bool memory_critical = health.memory_usage > config_.memory_threshold;
    bool disk_critical = health.disk_usage > config_.disk_threshold;
    
    if (cpu_critical || memory_critical || disk_critical) {
        return HealthStatus::UNHEALTHY;
    }
    
    // Check for degraded state (90% of threshold)
    bool cpu_high = health.cpu_usage > (config_.cpu_threshold * 0.9);
    bool memory_high = health.memory_usage > (config_.memory_threshold * 0.9);
    bool disk_high = health.disk_usage > (config_.disk_threshold * 0.9);
    
    if (cpu_high || memory_high || disk_high) {
        return HealthStatus::DEGRADED;
    }
    
    return HealthStatus::HEALTHY;
}

HealthStatus HealthMonitor::evaluate_cluster_health(const std::vector<NodeHealth>& nodes) const {
    if (nodes.empty()) {
        return HealthStatus::UNKNOWN;
    }
    
    int healthy = 0;
    int degraded = 0;
    int unhealthy = 0;
    
    for (const auto& node : nodes) {
        switch (node.status) {
            case HealthStatus::HEALTHY:
                healthy++;
                break;
            case HealthStatus::DEGRADED:
                degraded++;
                break;
            case HealthStatus::UNHEALTHY:
            case HealthStatus::UNKNOWN:
                unhealthy++;
                break;
        }
    }
    
    // Cluster is unhealthy if > 50% nodes are unhealthy
    if (unhealthy > nodes.size() / 2) {
        return HealthStatus::UNHEALTHY;
    }
    
    // Cluster is degraded if any node is degraded or unhealthy
    if (degraded > 0 || unhealthy > 0) {
        return HealthStatus::DEGRADED;
    }
    
    return HealthStatus::HEALTHY;
}

int64_t HealthMonitor::get_current_timestamp_ms() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

bool HealthMonitor::is_node_responsive(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        return false;
    }
    return it->second.health.is_reachable;
}

} // namespace jadevectordb
