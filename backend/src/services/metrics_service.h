#ifndef JADEVECTORDB_METRICS_SERVICE_H
#define JADEVECTORDB_METRICS_SERVICE_H

#include "lib/metrics.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <chrono>

namespace jadevectordb {

// Note: MetricType enum and Metric class are defined in lib/metrics.h
// We use those definitions here to avoid duplication

// Simple metric record for service-level tracking
struct MetricRecord {
    std::string name;
    std::string description;
    MetricType type;
    std::map<std::string, std::string> labels;
    double value;
    std::chrono::system_clock::time_point timestamp;

    MetricRecord() : type(MetricType::GAUGE), value(0.0),
                     timestamp(std::chrono::system_clock::now()) {}
    MetricRecord(const std::string& n, const std::string& desc, MetricType t)
        : name(n), description(desc), type(t), value(0.0),
          timestamp(std::chrono::system_clock::now()) {}
};

// Configuration for metrics collection
struct MetricsConfig {
    bool enabled = true;                           // Whether metrics collection is enabled
    int collection_interval_seconds = 5;          // How often to collect metrics
    int retention_hours = 24;                     // How long to retain metrics
    std::string export_format = "prometheus";     // "prometheus", "json", "csv"
    std::string export_path = "./metrics";        // Path to export metrics
    bool enable_system_metrics = true;           // Whether to collect system metrics
    bool enable_application_metrics = true;      // Whether to collect app-specific metrics
    bool enable_index_metrics = true;            // Whether to collect index-specific metrics
    bool enable_search_metrics = true;           // Whether to collect search-specific metrics
    bool enable_storage_metrics = true;          // Whether to collect storage metrics
    
    MetricsConfig() = default;
};

/**
 * @brief Service to aggregate and process collected metrics
 * 
 * This service handles the collection, aggregation, and export of various
 * system and application metrics for monitoring and observability.
 */
class MetricsService {
private:
    std::shared_ptr<logging::Logger> logger_;
    MetricsConfig config_;
    std::vector<MetricRecord> metrics_history_;         // Time-series history of metrics
    std::unordered_map<std::string, MetricRecord> current_metrics_;  // Current values by name
    mutable std::mutex metrics_mutex_;

    std::chrono::system_clock::time_point service_start_time_;
    
public:
    explicit MetricsService();
    ~MetricsService() = default;
    
    // Initialize the metrics service with configuration
    bool initialize(const MetricsConfig& config);
    
    // Record a new metric value
    Result<bool> record_metric(const std::string& name, 
                             double value, 
                             const std::unordered_map<std::string, std::string>& labels = {});
    
    // Record a counter metric (incremental)
    Result<bool> increment_counter(const std::string& name, 
                                 double increment = 1.0,
                                 const std::unordered_map<std::string, std::string>& labels = {});
    
    // Record a gauge metric
    Result<bool> set_gauge(const std::string& name, 
                         double value,
                         const std::unordered_map<std::string, std::string>& labels = {});
    
    // Record a histogram metric
    Result<bool> record_histogram(const std::string& name, 
                                double value,
                                const std::unordered_map<std::string, std::string>& labels = {});
    
    // Get current value of a metric
    Result<double> get_metric_value(const std::string& name) const;
    
    // Get current metric with all details
    Result<MetricRecord> get_metric(const std::string& name) const;
    
    // Get all current metrics
    Result<std::vector<MetricRecord>> get_all_metrics() const;
    
    // Get metrics by type
    Result<std::vector<MetricRecord>> get_metrics_by_type(MetricType type) const;
    
    // Get metrics by label
    Result<std::vector<MetricRecord>> get_metrics_by_label(const std::string& label_key, 
                                                   const std::string& label_value) const;
    
    // Get metrics by name pattern (supports simple wildcards)
    Result<std::vector<MetricRecord>> get_metrics_by_name_pattern(const std::string& pattern) const;
    
    // Export metrics in the configured format
    Result<std::string> export_metrics() const;
    
    // Export metrics to Prometheus format
    Result<std::string> export_prometheus_format() const;
    
    // Export metrics to JSON format
    Result<std::string> export_json_format() const;
    
    // Get metrics history for the last N minutes
    Result<std::vector<MetricRecord>> get_metrics_history_minutes(int minutes) const;
    
    // Get metrics history for the last N hours
    Result<std::vector<MetricRecord>> get_metrics_history_hours(int hours) const;
    
    // Clean up old metrics based on retention policy
    Result<bool> cleanup_old_metrics();
    
    // Update metrics configuration
    Result<bool> update_config(const MetricsConfig& new_config);
    
    // Get current metrics configuration
    MetricsConfig get_config() const;
    
    // Reset all metrics
    Result<bool> reset_metrics();

private:
    // Internal helper methods
    
    // Create a metric key for internal storage from name and labels
    std::string create_metric_key(const std::string& name, 
                                const std::unordered_map<std::string, std::string>& labels) const;
    
    // Validate metric name
    bool is_valid_metric_name(const std::string& name) const;
    
    // Validate metric labels
    bool are_valid_labels(const std::unordered_map<std::string, std::string>& labels) const;
    
    // Validate metric type
    bool is_valid_metric_type(MetricType type) const;
    
    // Apply retention policy to clean up old metrics
    void apply_retention_policy();
    
    // Format timestamp for export
    std::string format_timestamp(const std::chrono::system_clock::time_point& time) const;
    
    // Calculate percentiles for histogram metrics
    std::unordered_map<double, double> calculate_percentiles(const std::vector<double>& values) const;
    
    // Update the service start time (for uptime metrics)
    void set_service_start_time();
    
    // Generate standard system metrics
    Result<std::vector<MetricRecord>> generate_system_metrics() const;
    
    // Generate standard application metrics
    Result<std::vector<MetricRecord>> generate_application_metrics() const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_METRICS_SERVICE_H