#ifndef JADEVECTORDB_MONITORING_SERVICE_H
#define JADEVECTORDB_MONITORING_SERVICE_H

#include "lib/metrics.h"
#include "lib/logging.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>

namespace jadevectordb {

// System health status
struct SystemHealth {
    std::string status;              // "healthy", "degraded", "unhealthy"
    std::chrono::system_clock::time_point checked_at;
    std::unordered_map<std::string, std::string> components;  // component -> status
    std::unordered_map<std::string, float> metrics;          // metric -> value
    std::vector<std::string> issues;                         // list of issues if any
    int uptime_seconds;
    
    SystemHealth() : uptime_seconds(0) {}
};

// Status for a specific database
struct DatabaseStatus {
    std::string database_id;
    std::string status;              // "online", "offline", "degraded"
    size_t vector_count;
    size_t index_count;
    size_t storage_used_bytes;
    std::chrono::system_clock::time_point last_access;
    std::unordered_map<std::string, std::string> indexes_status;  // index_id -> status
    float query_performance_ms;      // Average query time
    float storage_utilization;       // Storage utilization percentage
    int uptime_seconds;
    
    DatabaseStatus() : vector_count(0), index_count(0), storage_used_bytes(0), 
                       query_performance_ms(0.0f), storage_utilization(0.0f), uptime_seconds(0) {}
};

// Configuration for monitoring
struct MonitoringConfig {
    bool enabled = true;                           // Whether monitoring is enabled
    int health_check_interval_seconds = 30;       // How often to perform health checks
    int metrics_collection_interval_seconds = 5;  // How often to collect metrics
    int status_update_interval_seconds = 10;      // How often to update status
    std::string log_level = "INFO";               // Log level for monitoring
    bool enable_prometheus_export = true;         // Whether to enable Prometheus metrics
    int prometheus_port = 9090;                   // Port for Prometheus metrics
    bool enable_alerts = true;                    // Whether to enable alerts
    std::vector<std::string> alert_channels;      // Where to send alerts ("log", "webhook", etc.)
    
    MonitoringConfig() = default;
};

/**
 * @brief Service to monitor system health and performance metrics
 * 
 * This service provides health checks, performance metrics collection,
 * and system status information for the vector database.
 */
class MonitoringService {
private:
    std::shared_ptr<logging::Logger> logger_;
    std::shared_ptr<MetricsService> metrics_service_;
    MonitoringConfig config_;
    SystemHealth current_health_;
    std::unordered_map<std::string, DatabaseStatus> database_statuses_;  // database_id -> status
    std::chrono::system_clock::time_point service_start_time_;
    std::mutex status_mutex_;
    
public:
    explicit MonitoringService(std::shared_ptr<MetricsService> metrics_service = nullptr);
    ~MonitoringService() = default;
    
    // Initialize the monitoring service with configuration
    bool initialize(const MonitoringConfig& config);
    
    // Perform system health check
    Result<SystemHealth> check_system_health() const;
    
    // Get status for a specific database
    Result<DatabaseStatus> get_database_status(const std::string& database_id) const;
    
    // Get status for all databases
    Result<std::vector<DatabaseStatus>> get_all_database_statuses() const;
    
    // Get overall system status
    Result<std::unordered_map<std::string, std::string>> get_system_status() const;
    
    // Collect and update metrics
    Result<bool> collect_metrics();
    
    // Get current metrics
    Result<std::unordered_map<std::string, std::string>> get_current_metrics() const;
    
    // Get metrics for a specific database
    Result<std::unordered_map<std::string, std::string>> get_database_metrics(
        const std::string& database_id) const;
    
    // Update database status
    Result<bool> update_database_status(const DatabaseStatus& db_status);
    
    // Check if system is healthy
    Result<bool> is_healthy() const;
    
    // Get service uptime in seconds
    int get_uptime_seconds() const;
    
    // Update monitoring configuration
    Result<bool> update_config(const MonitoringConfig& new_config);
    
    // Trigger alert if system is unhealthy
    Result<bool> trigger_alert(const std::string& alert_message);
    
    // Get the Prometheus metrics in text format
    Result<std::string> get_prometheus_metrics() const;
    
    // Get detailed system status including all components
    Result<std::unordered_map<std::string, std::string>> get_detailed_system_status() const;

private:
    // Internal helper methods
    
    // Perform detailed health checks
    SystemHealth perform_detailed_health_check() const;
    
    // Check if a specific component is healthy
    std::pair<bool, std::string> check_component_health(const std::string& component) const;
    
    // Calculate database-specific metrics
    DatabaseStatus calculate_database_status(const std::string& database_id) const;
    
    // Validate monitoring configuration
    bool validate_config(const MonitoringConfig& config) const;
    
    // Format metrics for Prometheus export
    std::string format_metrics_for_prometheus(const std::unordered_map<std::string, std::string>& metrics) const;
    
    // Send alert to configured channels
    Result<bool> send_alert_to_channels(const std::string& alert_message);
    
    // Collect resource usage metrics (CPU, memory, disk)
    std::unordered_map<std::string, std::string> collect_resource_metrics() const;
    
    // Collect performance metrics
    std::unordered_map<std::string, std::string> collect_performance_metrics() const;
    
    // Collect storage metrics
    std::unordered_map<std::string, std::string> collect_storage_metrics() const;
    
    // Calculate overall health status based on metrics
    std::string calculate_overall_health(const std::unordered_map<std::string, float>& metrics) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_MONITORING_SERVICE_H