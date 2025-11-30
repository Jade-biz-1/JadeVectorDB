#include "monitoring_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/statvfs.h>

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

// Helper function to get CPU usage from /proc/stat
double get_cpu_usage() {
    static unsigned long long prev_total = 0;
    static unsigned long long prev_idle = 0;

    std::ifstream stat_file("/proc/stat");
    if (!stat_file.is_open()) {
        return 0.0;
    }

    std::string line;
    std::getline(stat_file, line);
    stat_file.close();

    std::istringstream ss(line);
    std::string cpu;
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
    ss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

    unsigned long long idle_time = idle + iowait;
    unsigned long long total_time = user + nice + system + idle + iowait + irq + softirq + steal;

    unsigned long long total_diff = total_time - prev_total;
    unsigned long long idle_diff = idle_time - prev_idle;

    prev_total = total_time;
    prev_idle = idle_time;

    if (total_diff == 0) {
        return 0.0;
    }

    return 100.0 * (1.0 - static_cast<double>(idle_diff) / static_cast<double>(total_diff));
}

// Helper function to get memory usage from /proc/meminfo
double get_memory_usage() {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        return 0.0;
    }

    unsigned long long mem_total = 0;
    unsigned long long mem_available = 0;

    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            std::istringstream ss(line.substr(9));
            ss >> mem_total;
        } else if (line.find("MemAvailable:") == 0) {
            std::istringstream ss(line.substr(13));
            ss >> mem_available;
        }

        if (mem_total > 0 && mem_available > 0) {
            break;
        }
    }

    meminfo.close();

    if (mem_total == 0) {
        return 0.0;
    }

    return 100.0 * (1.0 - static_cast<double>(mem_available) / static_cast<double>(mem_total));
}

// Helper function to get disk usage
double get_disk_usage(const std::string& path = "/") {
    struct statvfs stat;
    if (statvfs(path.c_str(), &stat) != 0) {
        return 0.0;
    }

    unsigned long long total = stat.f_blocks * stat.f_frsize;
    unsigned long long available = stat.f_bavail * stat.f_frsize;

    if (total == 0) {
        return 0.0;
    }

    return 100.0 * (1.0 - static_cast<double>(available) / static_cast<double>(total));
}

Result<std::unordered_map<std::string, std::string>> MonitoringService::get_current_metrics() const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        std::unordered_map<std::string, std::string> metrics;

        // Collect real system metrics
        double cpu = get_cpu_usage();
        double memory = get_memory_usage();
        double disk = get_disk_usage();

        metrics["cpu_usage_percent"] = std::to_string(cpu);
        metrics["memory_usage_percent"] = std::to_string(memory);
        metrics["disk_usage_percent"] = std::to_string(disk);
        metrics["status"] = "healthy";

        LOG_DEBUG(logger_, "Generated current monitoring metrics: CPU=" +
                 std::to_string(cpu) + "%, Memory=" + std::to_string(memory) + "%");
        return metrics;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_current_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get current metrics: " + std::string(e.what()));
    }
}

Result<MonitoringMetrics> MonitoringService::get_monitoring_metrics() const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        MonitoringMetrics metrics;

        // Collect real system metrics (MonitoringMetrics has direct fields, not nested structs)
        metrics.cpu_usage_percent = get_cpu_usage();
        metrics.memory_usage_percent = get_memory_usage();
        metrics.disk_usage_percent = get_disk_usage();

        // Get connection metrics from metrics service if available
        if (metrics_service_) {
            auto conn_result = metrics_service_->get_metric("connections.active");
            if (conn_result.has_value()) {
                metrics.active_connections = static_cast<size_t>(conn_result.value().value);
            } else {
                metrics.active_connections = 0;  // Default value if not available
            }

            auto latency_result = metrics_service_->get_metric("query.latency.avg");
            if (latency_result.has_value()) {
                metrics.avg_query_latency_ms = latency_result.value().value;
            } else {
                metrics.avg_query_latency_ms = 0.0;  // Default value if not available
            }
        } else {
            metrics.active_connections = 0;
            metrics.avg_query_latency_ms = 0.0;
        }

        metrics.timestamp = std::chrono::system_clock::now();

        LOG_DEBUG(logger_, "Generated monitoring metrics struct");
        return metrics;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_monitoring_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get monitoring metrics: " + std::string(e.what()));
    }
}

Result<MonitoringAlert> MonitoringService::check_thresholds() const {
    try {
        auto metrics_result = get_monitoring_metrics();
        if (!metrics_result.has_value()) {
            LOG_ERROR(logger_, "Failed to get monitoring metrics for threshold checking");
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get metrics for threshold checking");
        }

        auto metrics = metrics_result.value();
        MonitoringAlert alert;
        alert.alert_id = "threshold_check_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        alert.severity = "info";
        alert.message = "Threshold check completed";
        alert.metric_name = "system_health";
        alert.threshold_value = 0.0;
        alert.actual_value = 0.0;
        alert.timestamp = std::chrono::system_clock::now();

        // Use configuration thresholds for checking
        if (metrics.cpu_usage_percent >= config_.system_thresholds.cpu_usage_critical) {
            alert.severity = "critical";
            alert.message = "CPU usage critical: " + std::to_string(metrics.cpu_usage_percent) + "% (threshold: " +
                           std::to_string(config_.system_thresholds.cpu_usage_critical) + "%)";
            alert.metric_name = "cpu_usage";
            alert.threshold_value = config_.system_thresholds.cpu_usage_critical;
            alert.actual_value = metrics.cpu_usage_percent;
        } else if (metrics.cpu_usage_percent >= config_.system_thresholds.cpu_usage_warning) {
            alert.severity = "warning";
            alert.message = "CPU usage high: " + std::to_string(metrics.cpu_usage_percent) + "% (threshold: " +
                           std::to_string(config_.system_thresholds.cpu_usage_warning) + "%)";
            alert.metric_name = "cpu_usage";
            alert.threshold_value = config_.system_thresholds.cpu_usage_warning;
            alert.actual_value = metrics.cpu_usage_percent;
        } else if (metrics.memory_usage_percent >= config_.system_thresholds.memory_usage_critical) {
            alert.severity = "critical";
            alert.message = "Memory usage critical: " + std::to_string(metrics.memory_usage_percent) + "% (threshold: " +
                           std::to_string(config_.system_thresholds.memory_usage_critical) + "%)";
            alert.metric_name = "memory_usage";
            alert.threshold_value = config_.system_thresholds.memory_usage_critical;
            alert.actual_value = metrics.memory_usage_percent;
        } else if (metrics.memory_usage_percent >= config_.system_thresholds.memory_usage_warning) {
            alert.severity = "warning";
            alert.message = "Memory usage high: " + std::to_string(metrics.memory_usage_percent) + "% (threshold: " +
                           std::to_string(config_.system_thresholds.memory_usage_warning) + "%)";
            alert.metric_name = "memory_usage";
            alert.threshold_value = config_.system_thresholds.memory_usage_warning;
            alert.actual_value = metrics.memory_usage_percent;
        } else if (metrics.disk_usage_percent >= config_.system_thresholds.disk_usage_critical) {
            alert.severity = "critical";
            alert.message = "Disk usage critical: " + std::to_string(metrics.disk_usage_percent) + "% (threshold: " +
                           std::to_string(config_.system_thresholds.disk_usage_critical) + "%)";
            alert.metric_name = "disk_usage";
            alert.threshold_value = config_.system_thresholds.disk_usage_critical;
            alert.actual_value = metrics.disk_usage_percent;
        } else if (metrics.disk_usage_percent >= config_.system_thresholds.disk_usage_warning) {
            alert.severity = "warning";
            alert.message = "Disk usage high: " + std::to_string(metrics.disk_usage_percent) + "% (threshold: " +
                           std::to_string(config_.system_thresholds.disk_usage_warning) + "%)";
            alert.metric_name = "disk_usage";
            alert.threshold_value = config_.system_thresholds.disk_usage_warning;
            alert.actual_value = metrics.disk_usage_percent;
        } else if (static_cast<double>(metrics.active_connections) >= config_.system_thresholds.connections_critical) {
            alert.severity = "critical";
            alert.message = "Active connections critical: " + std::to_string(metrics.active_connections) + " (threshold: " +
                           std::to_string(config_.system_thresholds.connections_critical) + ")";
            alert.metric_name = "active_connections";
            alert.threshold_value = static_cast<double>(config_.system_thresholds.connections_critical);
            alert.actual_value = static_cast<double>(metrics.active_connections);
        } else if (static_cast<double>(metrics.active_connections) >= config_.system_thresholds.connections_warning) {
            alert.severity = "warning";
            alert.message = "Active connections high: " + std::to_string(metrics.active_connections) + " (threshold: " +
                           std::to_string(config_.system_thresholds.connections_warning) + ")";
            alert.metric_name = "active_connections";
            alert.threshold_value = static_cast<double>(config_.system_thresholds.connections_warning);
            alert.actual_value = static_cast<double>(metrics.active_connections);
        } else if (metrics.avg_query_latency_ms >= config_.system_thresholds.query_latency_critical) {
            alert.severity = "critical";
            alert.message = "Query latency critical: " + std::to_string(metrics.avg_query_latency_ms) + "ms (threshold: " +
                           std::to_string(config_.system_thresholds.query_latency_critical) + "ms)";
            alert.metric_name = "query_latency";
            alert.threshold_value = config_.system_thresholds.query_latency_critical;
            alert.actual_value = metrics.avg_query_latency_ms;
        } else if (metrics.avg_query_latency_ms >= config_.system_thresholds.query_latency_warning) {
            alert.severity = "warning";
            alert.message = "Query latency high: " + std::to_string(metrics.avg_query_latency_ms) + "ms (threshold: " +
                           std::to_string(config_.system_thresholds.query_latency_warning) + "ms)";
            alert.metric_name = "query_latency";
            alert.threshold_value = config_.system_thresholds.query_latency_warning;
            alert.actual_value = metrics.avg_query_latency_ms;
        }

        LOG_DEBUG(logger_, "Threshold check completed: " + alert.message);
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
                                              const std::string& unit) {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        CustomMetric metric;
        metric.name = metric_name;
        metric.value = value;
        metric.unit = unit;
        metric.description = "";  // Optional description field
        metric.timestamp = std::chrono::system_clock::now();  // Fixed: use time_point directly

        custom_metrics_[metric_name] = metric;  // Now works because custom_metrics_ is a map

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
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get metrics for Prometheus export");
        }

        auto metrics = metrics_result.value();  // This is now a map<string, string>
        std::string prometheus_output;

        // Export system metrics from the map
        if (metrics.find("cpu_usage_percent") != metrics.end()) {
            prometheus_output += "# TYPE system_cpu_usage_percent gauge\n";
            prometheus_output += "system_cpu_usage_percent " + metrics["cpu_usage_percent"] + "\n\n";
        }

        if (metrics.find("memory_usage_percent") != metrics.end()) {
            prometheus_output += "# TYPE system_memory_usage_percent gauge\n";
            prometheus_output += "system_memory_usage_percent " + metrics["memory_usage_percent"] + "\n\n";
        }

        if (metrics.find("disk_usage_percent") != metrics.end()) {
            prometheus_output += "# TYPE system_disk_usage_percent gauge\n";
            prometheus_output += "system_disk_usage_percent " + metrics["disk_usage_percent"] + "\n\n";
        }

        // Export custom metrics
        for (const auto& [name, metric] : custom_metrics_) {
            prometheus_output += "# TYPE custom_" + name + " gauge\n";
            prometheus_output += "custom_" + name + " " + std::to_string(metric.value) + "\n\n";
        }

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
        metric.unit = "count";
        metric.description = "Node failure detected for node: " + node_id;
        metric.timestamp = std::chrono::system_clock::now();  // Fixed: use time_point

        custom_metrics_[metric.name] = metric;

        // Create an alert for the node failure
        MonitoringAlert alert;
        alert.alert_id = "node_failure_" + node_id + "_" +
                        std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        alert.severity = "critical";  // Changed from AlertSeverity::CRITICAL
        alert.message = "Node failure detected: " + node_id;
        alert.metric_name = "node_failure";
        alert.threshold_value = 0.0;
        alert.actual_value = 1.0;
        alert.timestamp = std::chrono::system_clock::now();  // Fixed: use time_point

        // Add to alerts list
        {
            std::lock_guard<std::mutex> lock2(alerts_mutex_);  // Renamed to avoid shadowing
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

                // Only log/store alerts with severity > "info"
                if (alert.severity == "warning" || alert.severity == "error" || alert.severity == "critical") {
                    // Add to alerts list
                    {
                        std::lock_guard<std::mutex> lock(alerts_mutex_);
                        alerts_.push_back(alert);

                        // Limit alerts list size
                        if (alerts_.size() > 1000) {
                            alerts_.erase(alerts_.begin(), alerts_.begin() + (alerts_.size() - 1000));
                        }
                    }

                    // Log the alert based on severity
                    if (alert.severity == "critical") {
                        LOG_ERROR(logger_, "[CRITICAL] " + alert.message);
                    } else if (alert.severity == "warning") {
                        LOG_WARN(logger_, "[WARNING] " + alert.message);
                    } else {
                        LOG_INFO(logger_, "[" + alert.severity + "] " + alert.message);
                    }
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
    LOG_DEBUG(logger_, "Validating monitoring configuration");

    // Validate basic configuration values
    if (config.health_check_interval_seconds <= 0) {
        LOG_ERROR(logger_, "Invalid health check interval: " + std::to_string(config.health_check_interval_seconds));
        return false;
    }

    if (config.metrics_collection_interval_seconds <= 0) {
        LOG_ERROR(logger_, "Invalid metrics collection interval: " + std::to_string(config.metrics_collection_interval_seconds));
        return false;
    }

    if (config.status_update_interval_seconds <= 0) {
        LOG_ERROR(logger_, "Invalid status update interval: " + std::to_string(config.status_update_interval_seconds));
        return false;
    }

    if (config.prometheus_port <= 0 || config.prometheus_port > 65535) {
        LOG_ERROR(logger_, "Invalid Prometheus port: " + std::to_string(config.prometheus_port));
        return false;
    }

    // Validate threshold values
    if (config.system_thresholds.cpu_usage_critical < config.system_thresholds.cpu_usage_warning ||
        config.system_thresholds.cpu_usage_warning < 0.0 || config.system_thresholds.cpu_usage_critical > 100.0) {
        LOG_ERROR(logger_, "Invalid CPU usage thresholds");
        return false;
    }

    if (config.system_thresholds.memory_usage_critical < config.system_thresholds.memory_usage_warning ||
        config.system_thresholds.memory_usage_warning < 0.0 || config.system_thresholds.memory_usage_critical > 100.0) {
        LOG_ERROR(logger_, "Invalid memory usage thresholds");
        return false;
    }

    if (config.system_thresholds.disk_usage_critical < config.system_thresholds.disk_usage_warning ||
        config.system_thresholds.disk_usage_warning < 0.0 || config.system_thresholds.disk_usage_critical > 100.0) {
        LOG_ERROR(logger_, "Invalid disk usage thresholds");
        return false;
    }

    if (config.system_thresholds.connections_critical < config.system_thresholds.connections_warning) {
        LOG_ERROR(logger_, "Invalid connection thresholds");
        return false;
    }

    if (config.system_thresholds.query_latency_critical < config.system_thresholds.query_latency_warning ||
        config.system_thresholds.query_latency_warning < 0.0 || config.system_thresholds.query_latency_critical < 0.0) {
        LOG_ERROR(logger_, "Invalid query latency thresholds");
        return false;
    }

    LOG_DEBUG(logger_, "Monitoring configuration validation passed");
    return true;
}

void MonitoringService::initialize_default_thresholds() {
    // Initialize default thresholds - these are now handled directly in the MonitoringConfig struct initialization
    LOG_DEBUG(logger_, "Using default monitoring thresholds from MonitoringConfig");

    // The thresholds are already initialized in the MonitoringConfig struct constructor
    // We just need to log these values
    LOG_INFO(logger_, "System thresholds - CPU: W=" + std::to_string(config_.system_thresholds.cpu_usage_warning) +
                          "/C=" + std::to_string(config_.system_thresholds.cpu_usage_critical));
    LOG_INFO(logger_, "System thresholds - Memory: W=" + std::to_string(config_.system_thresholds.memory_usage_warning) +
                          "/C=" + std::to_string(config_.system_thresholds.memory_usage_critical));
    LOG_INFO(logger_, "System thresholds - Disk: W=" + std::to_string(config_.system_thresholds.disk_usage_warning) +
                          "/C=" + std::to_string(config_.system_thresholds.disk_usage_critical));
    LOG_INFO(logger_, "System thresholds - Connections: W=" + std::to_string(config_.system_thresholds.connections_warning) +
                          "/C=" + std::to_string(config_.system_thresholds.connections_critical));
    LOG_INFO(logger_, "System thresholds - Latency: W=" + std::to_string(config_.system_thresholds.query_latency_warning) +
                          "/C=" + std::to_string(config_.system_thresholds.query_latency_critical));
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

// Note: The following methods are commented out because the required struct types
// (SystemMetrics, PerformanceMetrics, ClusterMetrics) are not yet defined.
// These can be uncommented and implemented when those structs are added.

/*
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
*/

} // namespace jadevectordb