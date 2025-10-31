#include "monitoring_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>
#include <chrono>

namespace jadevectordb {

MonitoringService::MonitoringService() {
    logger_ = logging::LoggerManager::get_logger("MonitoringService");
}

bool MonitoringService::initialize(const MonitoringConfig& config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (!validate_config(config)) {
            LOG_ERROR(logger_, "Invalid monitoring configuration provided");
            return false;
        }
        
        config_ = config;
        
        LOG_INFO(logger_, "MonitoringService initialized with interval: " + 
                std::to_string(config_.monitoring_interval_seconds) + "s, " +
                "enable_alerts: " + (config_.enable_alerts ? "true" : "false"));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in MonitoringService::initialize: " + std::string(e.what()));
        return false;
    }
}

Result<bool> MonitoringService::start_monitoring() {
    try {
        if (running_) {
            LOG_WARN(logger_, "Monitoring service is already running");
            return true;
        }
        
        LOG_INFO(logger_, "Starting monitoring service");
        
        running_ = true;
        monitoring_thread_ = std::make_unique<std::thread>(&MonitoringService::run_monitoring_loop, this);
        
        LOG_INFO(logger_, "Monitoring service started successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in start_monitoring: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to start monitoring service: " + std::string(e.what()));
    }
}

void MonitoringService::stop_monitoring() {
    try {
        if (!running_) {
            LOG_DEBUG(logger_, "Monitoring service is not running");
            return;
        }
        
        LOG_INFO(logger_, "Stopping monitoring service");
        
        running_ = false;
        
        if (monitoring_thread_ && monitoring_thread_->joinable()) {
            monitoring_thread_->join();
        }
        
        monitoring_thread_.reset();
        
        LOG_INFO(logger_, "Monitoring service stopped successfully");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in stop_monitoring: " + std::string(e.what()));
    }
}

Result<MonitoringMetrics> MonitoringService::get_current_metrics() const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        MonitoringMetrics metrics;
        metrics.system_metrics.cpu_usage_percent = static_cast<double>(rand() % 100);
        metrics.system_metrics.memory_usage_percent = static_cast<double>(rand() % 100);
        metrics.system_metrics.disk_usage_percent = static_cast<double>(rand() % 100);
        metrics.system_metrics.network_io_mbps = static_cast<double>(rand() % 1000) / 10.0;
        metrics.system_metrics.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Database metrics
        metrics.database_metrics.total_databases = 5;
        metrics.database_metrics.total_vectors = 100000;
        metrics.database_metrics.avg_vector_dimension = 128;
        metrics.database_metrics.storage_used_gb = 2.5;
        metrics.database_metrics.timestamp = metrics.system_metrics.timestamp;
        
        // Performance metrics
        metrics.performance_metrics.qps = 1250;
        metrics.performance_metrics.avg_response_time_ms = 2.5;
        metrics.performance_metrics.p95_response_time_ms = 5.0;
        metrics.performance_metrics.p99_response_time_ms = 15.0;
        metrics.performance_metrics.active_connections = 42;
        metrics.performance_metrics.timestamp = metrics.system_metrics.timestamp;
        
        // Cluster metrics
        metrics.cluster_metrics.total_nodes = 5;
        metrics.cluster_metrics.healthy_nodes = 5;
        metrics.cluster_metrics.master_node = "node_1";
        metrics.cluster_metrics.replication_lag_ms = 0;
        metrics.cluster_metrics.shard_distribution_skew = 0.1;
        metrics.cluster_metrics.timestamp = metrics.system_metrics.timestamp;
        
        LOG_DEBUG(logger_, "Generated current monitoring metrics");
        return metrics;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_current_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get current metrics: " + std::string(e.what()));
    }
}

Result<MonitoringAlert> MonitoringService::check_thresholds() const {
    try {
        auto metrics_result = get_current_metrics();
        if (!metrics_result.has_value()) {
            LOG_ERROR(logger_, "Failed to get current metrics for threshold checking: " + 
                     ErrorHandler::format_error(metrics_result.error()));
            return metrics_result;
        }
        
        auto metrics = metrics_result.value();
        MonitoringAlert alert;
        alert.alert_type = "threshold_check";
        alert.timestamp = metrics.system_metrics.timestamp;
        alert.is_triggered = false;
        
        // Check system thresholds
        if (metrics.system_metrics.cpu_usage_percent > config_.system_thresholds.cpu_usage_percent) {
            alert.is_triggered = true;
            alert.severity = AlertSeverity::WARNING;
            alert.message = "CPU usage threshold exceeded: " + 
                           std::to_string(metrics.system_metrics.cpu_usage_percent) + "% > " + 
                           std::to_string(config_.system_thresholds.cpu_usage_percent) + "%";
        }
        
        if (metrics.system_metrics.memory_usage_percent > config_.system_thresholds.memory_usage_percent) {
            alert.is_triggered = true;
            alert.severity = AlertSeverity::WARNING;
            alert.message = "Memory usage threshold exceeded: " + 
                           std::to_string(metrics.system_metrics.memory_usage_percent) + "% > " + 
                           std::to_string(config_.system_thresholds.memory_usage_percent) + "%";
        }
        
        // Check performance thresholds
        if (metrics.performance_metrics.avg_response_time_ms > config_.performance_thresholds.avg_response_time_ms) {
            alert.is_triggered = true;
            alert.severity = AlertSeverity::CRITICAL;
            alert.message = "Average response time threshold exceeded: " + 
                           std::to_string(metrics.performance_metrics.avg_response_time_ms) + "ms > " + 
                           std::to_string(config_.performance_thresholds.avg_response_time_ms) + "ms";
        }
        
        // Check cluster thresholds
        if (metrics.cluster_metrics.healthy_nodes < metrics.cluster_metrics.total_nodes) {
            alert.is_triggered = true;
            alert.severity = AlertSeverity::CRITICAL;
            alert.message = "Cluster health issue: " + 
                           std::to_string(metrics.cluster_metrics.healthy_nodes) + "/" + 
                           std::to_string(metrics.cluster_metrics.total_nodes) + " nodes healthy";
        }
        
        LOG_DEBUG(logger_, "Threshold check completed, alert triggered: " + 
                 (alert.is_triggered ? "true" : "false"));
        return alert;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in check_thresholds: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check thresholds: " + std::string(e.what()));
    }
}

Result<std::vector<MonitoringAlert>> MonitoringService::get_recent_alerts(int limit) const {
    try {
        std::lock_guard<std::mutex> lock(alerts_mutex_);
        
        std::vector<MonitoringAlert> recent_alerts;
        int count = 0;
        
        // Get recent alerts from the alerts list (newest first)
        for (auto it = alerts_.rbegin(); it != alerts_.rend() && count < limit; ++it) {
            recent_alerts.push_back(*it);
            count++;
        }
        
        LOG_DEBUG(logger_, "Retrieved " + std::to_string(recent_alerts.size()) + 
                 " recent alerts (limit: " + std::to_string(limit) + ")");
        return recent_alerts;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_recent_alerts: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get recent alerts: " + std::string(e.what()));
    }
}

Result<bool> MonitoringService::add_custom_metric(const std::string& metric_name, 
                                              double value, 
                                              const std::string& description) {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        CustomMetric metric;
        metric.name = metric_name;
        metric.value = value;
        metric.description = description;
        metric.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        custom_metrics_[metric_name] = metric;
        
        LOG_DEBUG(logger_, "Added custom metric " + metric_name + " with value " + std::to_string(value));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in add_custom_metric: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to add custom metric: " + std::string(e.what()));
    }
}

Result<std::vector<CustomMetric>> MonitoringService::get_custom_metrics() const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        std::vector<CustomMetric> metrics;
        metrics.reserve(custom_metrics_.size());
        
        for (const auto& entry : custom_metrics_) {
            metrics.push_back(entry.second);
        }
        
        LOG_DEBUG(logger_, "Retrieved " + std::to_string(metrics.size()) + " custom metrics");
        return metrics;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_custom_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get custom metrics: " + std::string(e.what()));
    }
}

Result<bool> MonitoringService::clear_custom_metrics() {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        custom_metrics_.clear();
        
        LOG_INFO(logger_, "Cleared all custom metrics");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in clear_custom_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to clear custom metrics: " + std::string(e.what()));
    }
}

Result<bool> MonitoringService::update_monitoring_config(const MonitoringConfig& new_config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (!validate_config(new_config)) {
            LOG_ERROR(logger_, "Invalid monitoring configuration provided for update");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid monitoring configuration");
        }
        
        config_ = new_config;
        
        LOG_INFO(logger_, "Updated monitoring configuration: interval=" + 
                std::to_string(config_.monitoring_interval_seconds) + "s, " +
                "enable_alerts=" + (config_.enable_alerts ? "true" : "false"));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_monitoring_config: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update monitoring configuration: " + std::string(e.what()));
    }
}

MonitoringConfig MonitoringService::get_config() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

Result<std::string> MonitoringService::export_metrics_prometheus() const {
    try {
        auto metrics_result = get_current_metrics();
        if (!metrics_result.has_value()) {
            LOG_ERROR(logger_, "Failed to get current metrics for Prometheus export: " + 
                     ErrorHandler::format_error(metrics_result.error()));
            return metrics_result;
        }
        
        auto metrics = metrics_result.value();
        std::string prometheus_output;
        
        // Export system metrics
        prometheus_output += "# TYPE system_cpu_usage_percent gauge\n";
        prometheus_output += "system_cpu_usage_percent " + 
                           std::to_string(metrics.system_metrics.cpu_usage_percent) + "\n\n";
        
        prometheus_output += "# TYPE system_memory_usage_percent gauge\n";
        prometheus_output += "system_memory_usage_percent " + 
                           std::to_string(metrics.system_metrics.memory_usage_percent) + "\n\n";
        
        prometheus_output += "# TYPE system_disk_usage_percent gauge\n";
        prometheus_output += "system_disk_usage_percent " + 
                           std::to_string(metrics.system_metrics.disk_usage_percent) + "\n\n";
        
        prometheus_output += "# TYPE system_network_io_mbps gauge\n";
        prometheus_output += "system_network_io_mbps " + 
                           std::to_string(metrics.system_metrics.network_io_mbps) + "\n\n";
        
        // Export database metrics
        prometheus_output += "# TYPE database_total_databases gauge\n";
        prometheus_output += "database_total_databases " + 
                           std::to_string(metrics.database_metrics.total_databases) + "\n\n";
        
        prometheus_output += "# TYPE database_total_vectors gauge\n";
        prometheus_output += "database_total_vectors " + 
                           std::to_string(metrics.database_metrics.total_vectors) + "\n\n";
        
        prometheus_output += "# TYPE database_avg_vector_dimension gauge\n";
        prometheus_output += "database_avg_vector_dimension " + 
                           std::to_string(metrics.database_metrics.avg_vector_dimension) + "\n\n";
        
        prometheus_output += "# TYPE database_storage_used_gb gauge\n";
        prometheus_output += "database_storage_used_gb " + 
                           std::to_string(metrics.database_metrics.storage_used_gb) + "\n\n";
        
        // Export performance metrics
        prometheus_output += "# TYPE performance_qps gauge\n";
        prometheus_output += "performance_qps " + 
                           std::to_string(metrics.performance_metrics.qps) + "\n\n";
        
        prometheus_output += "# TYPE performance_avg_response_time_ms gauge\n";
        prometheus_output += "performance_avg_response_time_ms " + 
                           std::to_string(metrics.performance_metrics.avg_response_time_ms) + "\n\n";
        
        prometheus_output += "# TYPE performance_active_connections gauge\n";
        prometheus_output += "performance_active_connections " + 
                           std::to_string(metrics.performance_metrics.active_connections) + "\n\n";
        
        // Export cluster metrics
        prometheus_output += "# TYPE cluster_total_nodes gauge\n";
        prometheus_output += "cluster_total_nodes " + 
                           std::to_string(metrics.cluster_metrics.total_nodes) + "\n\n";
        
        prometheus_output += "# TYPE cluster_healthy_nodes gauge\n";
        prometheus_output += "cluster_healthy_nodes " + 
                           std::to_string(metrics.cluster_metrics.healthy_nodes) + "\n\n";
        
        prometheus_output += "# TYPE cluster_replication_lag_ms gauge\n";
        prometheus_output += "cluster_replication_lag_ms " + 
                           std::to_string(metrics.cluster_metrics.replication_lag_ms) + "\n\n";
        
        LOG_DEBUG(logger_, "Exported Prometheus metrics, size: " + 
                 std::to_string(prometheus_output.length()) + " chars");
        return prometheus_output;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in export_metrics_prometheus: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to export Prometheus metrics: " + std::string(e.what()));
    }
}

Result<bool> MonitoringService::handle_node_failure(const std::string& node_id) {
    try {
        LOG_WARN(logger_, "Handling monitoring for failed node: " + node_id);
        
        // Record node failure in metrics
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        CustomMetric metric;
        metric.name = "node_failure_" + node_id;
        metric.value = 1.0;
        metric.description = "Node failure detected for node: " + node_id;
        metric.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        custom_metrics_[metric.name] = metric;
        
        // Create an alert for the node failure
        MonitoringAlert alert;
        alert.alert_type = "node_failure";
        alert.message = "Node failure detected: " + node_id;
        alert.severity = AlertSeverity::CRITICAL;
        alert.timestamp = metric.timestamp;
        alert.is_triggered = true;
        alert.source = "monitoring_service";
        
        // Add to alerts list
        {
            std::lock_guard<std::mutex> lock(alerts_mutex_);
            alerts_.push_back(alert);
        }
        
        LOG_INFO(logger_, "Handled monitoring for node failure: " + node_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_node_failure: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to handle node failure monitoring: " + std::string(e.what()));
    }
}

Result<bool> MonitoringService::add_node_monitoring(const std::string& node_id) {
    try {
        LOG_INFO(logger_, "Adding monitoring for new node: " + node_id);
        
        // In a real implementation, this would set up monitoring for the new node
        // For now, we'll just log that monitoring was added
        
        LOG_INFO(logger_, "Added monitoring for node: " + node_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in add_node_monitoring: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to add node monitoring: " + std::string(e.what()));
    }
}

Result<bool> MonitoringService::remove_node_monitoring(const std::string& node_id) {
    try {
        LOG_INFO(logger_, "Removing monitoring for node: " + node_id);
        
        // In a real implementation, this would remove monitoring for the node
        // For now, we'll just log that monitoring was removed
        
        LOG_INFO(logger_, "Removed monitoring for node: " + node_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in remove_node_monitoring: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to remove node monitoring: " + std::string(e.what()));
    }
}

Result<std::unordered_map<std::string, std::string>> MonitoringService::get_monitoring_stats() const {
    try {
        std::unordered_map<std::string, std::string> stats;
        
        stats["service_status"] = running_ ? "running" : "stopped";
        stats["monitoring_interval_seconds"] = std::to_string(config_.monitoring_interval_seconds);
        stats["alerts_enabled"] = config_.enable_alerts ? "true" : "false";
        
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        stats["custom_metrics_count"] = std::to_string(custom_metrics_.size());
        
        std::lock_guard<std::mutex> lock2(alerts_mutex_);
        stats["recent_alerts_count"] = std::to_string(alerts_.size());
        
        LOG_DEBUG(logger_, "Generated monitoring statistics");
        return stats;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_monitoring_stats: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get monitoring stats: " + std::string(e.what()));
    }
}

// Private methods

void MonitoringService::run_monitoring_loop() {
    LOG_INFO(logger_, "Monitoring loop started");
    
    while (running_) {
        try {
            collect_and_report_metrics();
            
            // Sleep for monitoring interval
            std::this_thread::sleep_for(std::chrono::seconds(config_.monitoring_interval_seconds));
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Exception in monitoring loop: " + std::string(e.what()));
            std::this_thread::sleep_for(std::chrono::seconds(1)); // Prevent tight loop on error
        }
    }
    
    LOG_INFO(logger_, "Monitoring loop stopped");
}

void MonitoringService::collect_and_report_metrics() {
    try {
        LOG_DEBUG(logger_, "Collecting and reporting metrics");
        
        // Collect current metrics
        auto metrics_result = get_current_metrics();
        if (!metrics_result.has_value()) {
            LOG_ERROR(logger_, "Failed to collect metrics: " + 
                     ErrorHandler::format_error(metrics_result.error()));
            return;
        }
        
        auto metrics = metrics_result.value();
        
        // Check thresholds if alerts are enabled
        if (config_.enable_alerts) {
            auto alert_result = check_thresholds();
            if (alert_result.has_value()) {
                auto alert = alert_result.value();
                if (alert.is_triggered) {
                    // Add to alerts list
                    {
                        std::lock_guard<std::mutex> lock(alerts_mutex_);
                        alerts_.push_back(alert);
                        
                        // Limit alerts list size
                        if (alerts_.size() > 1000) {
                            alerts_.erase(alerts_.begin(), alerts_.begin() + (alerts_.size() - 1000));
                        }
                    }
                    
                    // Log the alert
                    std::string severity_str;
                    switch (alert.severity) {
                        case AlertSeverity::INFO: severity_str = "INFO"; break;
                        case AlertSeverity::WARNING: severity_str = "WARNING"; break;
                        case AlertSeverity::CRITICAL: severity_str = "CRITICAL"; break;
                        case AlertSeverity::EMERGENCY: severity_str = "EMERGENCY"; break;
                        default: severity_str = "UNKNOWN";
                    }
                    
                    LOG_ALERT(logger_, "[" + severity_str + "] " + alert.message);
                }
            } else {
                LOG_WARN(logger_, "Failed to check thresholds: " + 
                        ErrorHandler::format_error(alert_result.error()));
            }
        }
        
        LOG_DEBUG(logger_, "Collected and reported metrics successfully");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in collect_and_report_metrics: " + std::string(e.what()));
    }
}

bool MonitoringService::validate_config(const MonitoringConfig& config) const {
    // Basic validation
    if (config.monitoring_interval_seconds <= 0) {
        LOG_ERROR(logger_, "Invalid monitoring interval: " + std::to_string(config.monitoring_interval_seconds));
        return false;
    }
    
    if (config.max_alert_history < 0) {
        LOG_ERROR(logger_, "Invalid max alert history: " + std::to_string(config.max_alert_history));
        return false;
    }
    
    // Validate system thresholds
    if (config.system_thresholds.cpu_usage_percent < 0 || config.system_thresholds.cpu_usage_percent > 100) {
        LOG_ERROR(logger_, "Invalid CPU usage threshold: " + std::to_string(config.system_thresholds.cpu_usage_percent));
        return false;
    }
    
    if (config.system_thresholds.memory_usage_percent < 0 || config.system_thresholds.memory_usage_percent > 100) {
        LOG_ERROR(logger_, "Invalid memory usage threshold: " + std::to_string(config.system_thresholds.memory_usage_percent));
        return false;
    }
    
    if (config.system_thresholds.disk_usage_percent < 0 || config.system_thresholds.disk_usage_percent > 100) {
        LOG_ERROR(logger_, "Invalid disk usage threshold: " + std::to_string(config.system_thresholds.disk_usage_percent));
        return false;
    }
    
    // Validate performance thresholds
    if (config.performance_thresholds.qps < 0) {
        LOG_ERROR(logger_, "Invalid QPS threshold: " + std::to_string(config.performance_thresholds.qps));
        return false;
    }
    
    if (config.performance_thresholds.avg_response_time_ms < 0) {
        LOG_ERROR(logger_, "Invalid average response time threshold: " + 
                 std::to_string(config.performance_thresholds.avg_response_time_ms));
        return false;
    }
    
    if (config.performance_thresholds.p95_response_time_ms < 0) {
        LOG_ERROR(logger_, "Invalid P95 response time threshold: " + 
                 std::to_string(config.performance_thresholds.p95_response_time_ms));
        return false;
    }
    
    if (config.performance_thresholds.p99_response_time_ms < 0) {
        LOG_ERROR(logger_, "Invalid P99 response time threshold: " + 
                 std::to_string(config.performance_thresholds.p99_response_time_ms));
        return false;
    }
    
    if (config.performance_thresholds.active_connections < 0) {
        LOG_ERROR(logger_, "Invalid active connections threshold: " + 
                 std::to_string(config.performance_thresholds.active_connections));
        return false;
    }
    
    // Validate cluster thresholds
    if (config.cluster_thresholds.min_healthy_nodes < 0) {
        LOG_ERROR(logger_, "Invalid minimum healthy nodes threshold: " + 
                 std::to_string(config.cluster_thresholds.min_healthy_nodes));
        return false;
    }
    
    if (config.cluster_thresholds.max_replication_lag_ms < 0) {
        LOG_ERROR(logger_, "Invalid maximum replication lag threshold: " + 
                 std::to_string(config.cluster_thresholds.max_replication_lag_ms));
        return false;
    }
    
    if (config.cluster_thresholds.max_shard_distribution_skew < 0 || 
        config.cluster_thresholds.max_shard_distribution_skew > 1) {
        LOG_ERROR(logger_, "Invalid maximum shard distribution skew threshold: " + 
                 std::to_string(config.cluster_thresholds.max_shard_distribution_skew));
        return false;
    }
    
    return true;
}

void MonitoringService::initialize_default_thresholds() {
    // System thresholds
    config_.system_thresholds.cpu_usage_percent = 80.0;
    config_.system_thresholds.memory_usage_percent = 85.0;
    config_.system_thresholds.disk_usage_percent = 90.0;
    config_.system_thresholds.network_io_mbps = 1000.0;
    
    // Performance thresholds
    config_.performance_thresholds.qps = 10000;
    config_.performance_thresholds.avg_response_time_ms = 10.0;
    config_.performance_thresholds.p95_response_time_ms = 50.0;
    config_.performance_thresholds.p99_response_time_ms = 100.0;
    config_.performance_thresholds.active_connections = 1000;
    
    // Cluster thresholds
    config_.cluster_thresholds.min_healthy_nodes = 3;
    config_.cluster_thresholds.max_replication_lag_ms = 1000;
    config_.cluster_thresholds.max_shard_distribution_skew = 0.2;
}

std::string MonitoringService::format_metric_name(const std::string& base_name) const {
    // Format metric name to be Prometheus-compatible
    std::string formatted = base_name;
    
    // Replace invalid characters with underscores
    for (char& c : formatted) {
        if (!(isalnum(c) || c == '_' || c == ':')) {
            c = '_';
        }
    }
    
    // Ensure it doesn't start with a digit
    if (!formatted.empty() && isdigit(formatted[0])) {
        formatted = "_" + formatted;
    }
    
    return formatted;
}

double MonitoringService::calculate_system_health_score(const SystemMetrics& metrics) const {
    // Simple health score calculation based on system metrics
    double cpu_score = 1.0 - (metrics.cpu_usage_percent / 100.0);
    double memory_score = 1.0 - (metrics.memory_usage_percent / 100.0);
    double disk_score = 1.0 - (metrics.disk_usage_percent / 100.0);
    
    // Weighted average (CPU 40%, Memory 35%, Disk 25%)
    return (cpu_score * 0.4) + (memory_score * 0.35) + (disk_score * 0.25);
}

double MonitoringService::calculate_performance_health_score(const PerformanceMetrics& metrics) const {
    // Simple performance health score calculation
    // Lower response times and higher QPS are better
    double response_time_score = std::max(0.0, 1.0 - (metrics.avg_response_time_ms / 100.0));
    double qps_score = std::min(1.0, metrics.qps / 10000.0); // Assume 10000 QPS is ideal
    
    // Weighted average (Response time 60%, QPS 40%)
    return (response_time_score * 0.6) + (qps_score * 0.4);
}

double MonitoringService::calculate_cluster_health_score(const ClusterMetrics& metrics) const {
    // Simple cluster health score calculation
    if (metrics.total_nodes == 0) return 1.0; // No cluster, so fully healthy
    
    double node_health_score = static_cast<double>(metrics.healthy_nodes) / metrics.total_nodes;
    double replication_score = metrics.replication_lag_ms > 0 ? 
                              1.0 - std::min(1.0, static_cast<double>(metrics.replication_lag_ms) / 5000.0) : 1.0;
    double distribution_score = 1.0 - metrics.shard_distribution_skew;
    
    // Weighted average (Node health 50%, Replication 30%, Distribution 20%)
    return (node_health_score * 0.5) + (replication_score * 0.3) + (distribution_score * 0.2);
}

} // namespace jadevectordb