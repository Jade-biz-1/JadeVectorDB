#ifndef JADEVECTORDB_MONITORING_SERVICE_H
#define JADEVECTORDB_MONITORING_SERVICE_H

#include "lib/metrics.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include "services/metrics_service.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <thread>

namespace jadevectordb {

// Supporting structures for monitoring (forward declarations)
struct MonitoringMetrics {
    double cpu_usage_percent = 0.0;
    double memory_usage_percent = 0.0;
    double disk_usage_percent = 0.0;
    size_t active_connections = 0;
    double avg_query_latency_ms = 0.0;
    std::unordered_map<std::string, double> custom_metrics;
    std::chrono::system_clock::time_point timestamp;
};

struct MonitoringAlert {
    std::string alert_id;
    std::string severity;  // "warning", "error", "critical"
    std::string message;
    std::string metric_name;
    double threshold_value;
    double actual_value;
    std::chrono::system_clock::time_point timestamp;
};

struct CustomMetric {
    std::string name;
    double value;
    std::string unit;
    std::string description;  // Added for add_custom_metric()
    std::chrono::system_clock::time_point timestamp;
};

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
    int monitoring_interval_seconds = 5;          // General monitoring interval (alias for metrics_collection)
    int status_update_interval_seconds = 10;      // How often to update status
    std::string log_level = "INFO";               // Log level for monitoring
    bool enable_prometheus_export = true;         // Whether to enable Prometheus metrics
    int prometheus_port = 9090;                   // Port for Prometheus metrics
    bool enable_alerts = true;                    // Whether to enable alerts
    std::vector<std::string> alert_channels;      // Where to send alerts ("log", "webhook", etc.)

    // Threshold configurations
    struct ThresholdConfig {
        double cpu_usage_critical = 90.0;         // CPU usage percentage for critical alert
        double cpu_usage_warning = 75.0;          // CPU usage percentage for warning
        double memory_usage_critical = 90.0;      // Memory usage percentage for critical alert
        double memory_usage_warning = 75.0;       // Memory usage percentage for warning
        double disk_usage_critical = 95.0;        // Disk usage percentage for critical alert
        double disk_usage_warning = 85.0;         // Disk usage percentage for warning
        size_t connections_critical = 1000;       // Max connections for critical alert
        size_t connections_warning = 800;         // Max connections for warning
        double query_latency_critical = 1000.0;   // Query latency in ms for critical alert
        double query_latency_warning = 500.0;     // Query latency in ms for warning
    } system_thresholds;

    struct PerformanceThresholds {
        double index_build_time_critical = 300.0;  // Index build time in seconds for critical alert
        double index_build_time_warning = 120.0;   // Index build time in seconds for warning
        double query_throughput_critical = 100.0;  // Queries per second for critical alert (low)
        double query_throughput_warning = 500.0;   // Queries per second for warning (low)
    } performance_thresholds;

    struct ClusterThresholds {
        int node_failure_critical = 2;              // Number of failed nodes for critical alert
        int node_failure_warning = 1;               // Number of failed nodes for warning
        double replication_lag_critical = 10.0;     // Replication lag in seconds for critical
        double replication_lag_warning = 5.0;       // Replication lag in seconds for warning
        double cluster_latency_critical = 2000.0;   // Cluster latency in ms for critical
        double cluster_latency_warning = 1000.0;    // Cluster latency in ms for warning
    } cluster_thresholds;

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
    mutable std::mutex status_mutex_;
    mutable std::mutex config_mutex_;
    bool running_ = false;
    std::unique_ptr<std::thread> monitoring_thread_;

public:
    MonitoringService();
    explicit MonitoringService(std::shared_ptr<MetricsService> metrics_service);
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

    // Get current monitoring configuration
    MonitoringConfig get_config() const;

    // Trigger alert if system is unhealthy
    Result<bool> trigger_alert(const std::string& alert_message);
    
    // Get the Prometheus metrics in text format
    Result<std::string> get_prometheus_metrics() const;

    // Get detailed system status including all components
    Result<std::unordered_map<std::string, std::string>> get_detailed_system_status() const;

    // Start monitoring service
    Result<bool> start_monitoring();

    // Stop monitoring service
    void stop_monitoring();

    // Additional monitoring methods from implementation
    Result<MonitoringMetrics> get_monitoring_metrics() const;
    Result<MonitoringAlert> check_thresholds() const;
    Result<std::vector<MonitoringAlert>> get_recent_alerts(int limit = 100) const;
    Result<bool> add_custom_metric(const std::string& metric_name, double value, const std::string& unit = "");
    Result<std::vector<CustomMetric>> get_custom_metrics() const;
    Result<bool> clear_custom_metrics();
    Result<bool> update_monitoring_config(const MonitoringConfig& new_config);
    Result<std::string> export_metrics_prometheus() const;
    Result<bool> handle_node_failure(const std::string& node_id);
    Result<bool> add_node_monitoring(const std::string& node_id);
    Result<bool> remove_node_monitoring(const std::string& node_id);
    Result<std::unordered_map<std::string, std::string>> get_monitoring_stats() const;

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

    // Monitoring thread method
    void run_monitoring_loop();

    // Collect and report metrics
    void collect_and_report_metrics();

    // Initialize default thresholds
    void initialize_default_thresholds();

    // Format metric name
    std::string format_metric_name(const std::string& base_name) const;

    // Additional private members
    mutable std::mutex metrics_mutex_;
    mutable std::mutex alerts_mutex_;  // Added for thread-safe alert access
    std::vector<MonitoringAlert> recent_alerts_;
    std::vector<MonitoringAlert> alerts_;  // Added for get_recent_alerts()
    std::unordered_map<std::string, CustomMetric> custom_metrics_;  // Changed from vector to map
    std::unordered_map<std::string, std::string> monitored_nodes_;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_MONITORING_SERVICE_H