#ifndef JADEVECTORDB_ANALYTICS_DASHBOARD_H
#define JADEVECTORDB_ANALYTICS_DASHBOARD_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <atomic>

#include "lib/metrics.h"
#include "services/monitoring_service.h"
#include "services/metrics_service.h"
#include "lib/logging.h"

namespace jadevectordb {
namespace analytics {

    // Enum for different chart types
    enum class ChartType {
        LINE_CHART,
        BAR_CHART,
        PIE_CHART,
        HEATMAP,
        SCATTER_PLOT,
        GAUGE,
        TABLE
    };

    // Structure for dashboard widget configuration
    struct WidgetConfig {
        std::string id;
        std::string title;
        std::string description;
        ChartType chart_type;
        std::string metric_query;  // Query to fetch metrics
        std::unordered_map<std::string, std::string> parameters;  // Additional parameters
        int refresh_interval_seconds;  // How often to refresh the widget
        std::vector<std::string> tags;  // Tags for categorization
        bool enabled;  // Whether the widget is enabled
        std::string size_class;  // Size class (small, medium, large, full-width)
        int row_span;
        int col_span;
        int position_x;  // Grid position
        int position_y;
        
        WidgetConfig() : refresh_interval_seconds(30), enabled(true), 
                        row_span(1), col_span(1), position_x(0), position_y(0) {}
    };

    // Structure for dashboard layout
    struct DashboardLayout {
        std::string id;
        std::string name;
        std::string description;
        std::vector<WidgetConfig> widgets;
        int grid_columns;  // Number of columns in the grid layout
        std::string theme;  // Dashboard theme
        bool is_default;  // Whether this is the default dashboard
        std::chrono::system_clock::time_point created_at;
        std::chrono::system_clock::time_point updated_at;
        
        DashboardLayout() : grid_columns(12), is_default(false) {}
    };

    // Structure for real-time metric data
    struct MetricDataPoint {
        std::string metric_name;
        double value;
        std::chrono::system_clock::time_point timestamp;
        std::unordered_map<std::string, std::string> labels;
        std::string unit;  // Unit of measurement (e.g., "ms", "bytes", "%")
        
        MetricDataPoint() : value(0.0) {}
    };

    // Structure for query pattern analysis
    struct QueryPattern {
        std::string query_type;
        std::string database_id;
        int frequency;
        double avg_response_time_ms;
        double median_response_time_ms;
        double p95_response_time_ms;
        double p99_response_time_ms;
        std::chrono::system_clock::time_point first_seen;
        std::chrono::system_clock::time_point last_seen;
        std::vector<std::string> common_terms;  // For search queries
        std::unordered_map<std::string, std::string> filters;  // Common filters used
        int error_count;
        double success_rate;  // Percentage of successful queries
        
        QueryPattern() : frequency(0), avg_response_time_ms(0.0), median_response_time_ms(0.0),
                        p95_response_time_ms(0.0), p99_response_time_ms(0.0), error_count(0), success_rate(0.0) {}
    };

    // Structure for resource utilization data
    struct ResourceUtilization {
        std::string resource_type;  // "cpu", "memory", "disk", "network"
        std::string resource_id;     // Identifier for the specific resource
        double utilization_percentage;
        double used_amount;
        double total_amount;
        std::string unit;             // "bytes", "cores", etc.
        std::chrono::system_clock::time_point measured_at;
        std::vector<double> history;   // Historical utilization values
        std::string status;          // "normal", "warning", "critical"
        std::string recommendation;   // Recommendation based on utilization
        
        ResourceUtilization() : utilization_percentage(0.0), used_amount(0.0), total_amount(0.0) {}
    };

    // Structure for anomaly detection
    struct Anomaly {
        std::string id;
        std::string metric_name;
        std::string anomaly_type;     // "spike", "drop", "trend", "seasonal"
        double current_value;
        double expected_value;
        double deviation_percentage;
        std::chrono::system_clock::time_point detected_at;
        std::chrono::system_clock::time_point resolved_at;
        std::string severity;         // "low", "medium", "high", "critical"
        std::string description;
        std::vector<std::string> affected_components;
        std::string resolution_status;  // "detected", "investigating", "resolved", "acknowledged"
        std::string assigned_to;         // User or team assigned to investigate
        std::vector<std::string> suggestions;  // Suggestions for resolving the anomaly
        
        Anomaly() : current_value(0.0), expected_value(0.0), deviation_percentage(0.0) {}
    };

    // Structure for alert configuration
    struct AlertConfig {
        std::string id;
        std::string name;
        std::string description;
        std::string metric_name;
        std::string condition;        // ">", "<", "==", ">=", "<=", "!="
        double threshold;
        std::string severity;          // "info", "warning", "error", "critical"
        int cooldown_period_seconds;  // Minimum time between alerts
        std::vector<std::string> notification_channels;  // "email", "slack", "webhook", etc.
        std::unordered_map<std::string, std::string> notification_params;  // Channel-specific parameters
        bool enabled;
        std::chrono::system_clock::time_point created_at;
        std::chrono::system_clock::time_point last_triggered;
        int trigger_count;
        
        AlertConfig() : threshold(0.0), cooldown_period_seconds(300), enabled(true), trigger_count(0) {}
    };

    // Callback function type for real-time updates
    using DashboardUpdateCallback = std::function<void(const std::string& widget_id, 
                                                      const std::vector<MetricDataPoint>& data)>;

    /**
     * @brief Interface for real-time metric data providers
     * 
     * This interface defines how the dashboard retrieves real-time metric data
     */
    class IMetricDataProvider {
    public:
        virtual ~IMetricDataProvider() = default;
        
        /**
         * @brief Fetch real-time metric data
         * @param query Query to fetch specific metrics
         * @param start_time Start time for time-series data
         * @param end_time End time for time-series data
         * @param limit Maximum number of data points to return
         * @return Vector of metric data points
         */
        virtual std::vector<MetricDataPoint> fetch_metrics(
            const std::string& query,
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time,
            int limit = 1000) = 0;
        
        /**
         * @brief Subscribe to real-time metric updates
         * @param query Query to subscribe to
         * @param callback Function to call when updates are available
         * @return Subscription ID
         */
        virtual std::string subscribe_to_updates(
            const std::string& query,
            DashboardUpdateCallback callback) = 0;
        
        /**
         * @brief Unsubscribe from real-time metric updates
         * @param subscription_id ID of the subscription to cancel
         */
        virtual void unsubscribe_from_updates(const std::string& subscription_id) = 0;
        
        /**
         * @brief Get available metrics
         * @return Vector of available metric names
         */
        virtual std::vector<std::string> get_available_metrics() const = 0;
    };

    /**
     * @brief Interface for query pattern analysis
     * 
     * This interface provides query pattern analysis capabilities
     */
    class IQueryPatternAnalyzer {
    public:
        virtual ~IQueryPatternAnalyzer() = default;
        
        /**
         * @brief Analyze query patterns over a time period
         * @param database_id Database to analyze (empty for all)
         * @param start_time Start time for analysis
         * @param end_time End time for analysis
         * @return Vector of query patterns
         */
        virtual std::vector<QueryPattern> analyze_query_patterns(
            const std::string& database_id,
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time) = 0;
        
        /**
         * @brief Get most frequent queries
         * @param database_id Database to analyze
         * @param limit Maximum number of queries to return
         * @param time_window Time window to consider
         * @return Vector of frequent queries
         */
        virtual std::vector<QueryPattern> get_frequent_queries(
            const std::string& database_id = "",
            int limit = 10,
            std::chrono::hours time_window = std::chrono::hours(24)) = 0;
        
        /**
         * @brief Get slowest queries
         * @param database_id Database to analyze
         * @param limit Maximum number of queries to return
         * @param time_window Time window to consider
         * @return Vector of slow queries
         */
        virtual std::vector<QueryPattern> get_slowest_queries(
            const std::string& database_id = "",
            int limit = 10,
            std::chrono::hours time_window = std::chrono::hours(24)) = 0;
        
        /**
         * @brief Get queries with highest error rates
         * @param database_id Database to analyze
         * @param limit Maximum number of queries to return
         * @param time_window Time window to consider
         * @return Vector of error-prone queries
         */
        virtual std::vector<QueryPattern> get_error_prone_queries(
            const std::string& database_id = "",
            int limit = 10,
            std::chrono::hours time_window = std::chrono::hours(24)) = 0;
    };

    /**
     * @brief Interface for resource utilization monitoring
     * 
     * This interface provides resource utilization data and heatmaps
     */
    class IResourceMonitor {
    public:
        virtual ~IResourceMonitor() = default;
        
        /**
         * @brief Get current resource utilization
         * @param resource_type Type of resource to monitor
         * @return Vector of resource utilization data
         */
        virtual std::vector<ResourceUtilization> get_current_utilization(
            const std::string& resource_type = "") = 0;
        
        /**
         * @brief Get resource utilization history
         * @param resource_type Type of resource to monitor
         * @param start_time Start time for history
         * @param end_time End time for history
         * @return Vector of resource utilization data points
         */
        virtual std::vector<ResourceUtilization> get_utilization_history(
            const std::string& resource_type,
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time) = 0;
        
        /**
         * @brief Get resource utilization heatmap
         * @param resource_type Type of resource to monitor
         * @param time_granularity Granularity of time buckets (minutes)
         * @param start_time Start time for heatmap
         * @param end_time End time for heatmap
         * @return 2D vector representing utilization heatmap
         */
        virtual std::vector<std::vector<double>> get_utilization_heatmap(
            const std::string& resource_type,
            int time_granularity_minutes,
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time) = 0;
        
        /**
         * @brief Get resource recommendations based on utilization trends
         * @return Vector of resource utilization recommendations
         */
        virtual std::vector<ResourceUtilization> get_recommendations() = 0;
    };

    /**
     * @brief Interface for anomaly detection
     * 
     * This interface provides anomaly detection and alerting capabilities
     */
    class IAnomalyDetector {
    public:
        virtual ~IAnomalyDetector() = default;
        
        /**
         * @brief Detect anomalies in metrics
         * @param metric_name Name of the metric to analyze
         * @param start_time Start time for analysis
         * @param end_time End time for analysis
         * @return Vector of detected anomalies
         */
        virtual std::vector<Anomaly> detect_anomalies(
            const std::string& metric_name,
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time) = 0;
        
        /**
         * @brief Get current active anomalies
         * @return Vector of active anomalies
         */
        virtual std::vector<Anomaly> get_active_anomalies() = 0;
        
        /**
         * @brief Acknowledge an anomaly
         * @param anomaly_id ID of the anomaly to acknowledge
         * @param user User acknowledging the anomaly
         * @return True if successful
         */
        virtual bool acknowledge_anomaly(const std::string& anomaly_id, const std::string& user) = 0;
        
        /**
         * @brief Resolve an anomaly
         * @param anomaly_id ID of the anomaly to resolve
         * @param user User resolving the anomaly
         * @return True if successful
         */
        virtual bool resolve_anomaly(const std::string& anomaly_id, const std::string& user) = 0;
        
        /**
         * @brief Configure alerts for anomalies
         * @param config Alert configuration
         * @return True if successful
         */
        virtual bool configure_alert(const AlertConfig& config) = 0;
        
        /**
         * @brief Get triggered alerts
         * @param start_time Start time for alerts
         * @param end_time End time for alerts
         * @return Vector of triggered alerts
         */
        virtual std::vector<AlertConfig> get_triggered_alerts(
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time) = 0;
    };

    /**
     * @brief Advanced Analytics Dashboard
     * 
     * This class provides a comprehensive analytics dashboard with real-time metrics,
     * query pattern analysis, resource utilization monitoring, and anomaly detection.
     */
    class AnalyticsDashboard {
    private:
        std::shared_ptr<logging::Logger> logger_;
        std::unique_ptr<IMetricDataProvider> metric_data_provider_;
        std::unique_ptr<IQueryPatternAnalyzer> query_analyzer_;
        std::unique_ptr<IResourceMonitor> resource_monitor_;
        std::unique_ptr<IAnomalyDetector> anomaly_detector_;
        
        std::unordered_map<std::string, DashboardLayout> layouts_;
        std::unordered_map<std::string, AlertConfig> alert_configs_;
        std::vector<Anomaly> active_anomalies_;
        
        mutable std::shared_mutex dashboard_mutex_;
        mutable std::mutex alert_mutex_;
        
        std::atomic<bool> running_;
        std::thread background_thread_;
        std::chrono::seconds update_interval_;
        
    public:
        explicit AnalyticsDashboard(
            std::unique_ptr<IMetricDataProvider> metric_provider = nullptr,
            std::unique_ptr<IQueryPatternAnalyzer> query_analyzer = nullptr,
            std::unique_ptr<IResourceMonitor> resource_monitor = nullptr,
            std::unique_ptr<IAnomalyDetector> anomaly_detector = nullptr
        );
        
        ~AnalyticsDashboard();
        
        /**
         * @brief Initialize the dashboard with configuration
         * @param update_interval_seconds How often to update dashboard data
         * @return True if initialization was successful
         */
        bool initialize(int update_interval_seconds = 30);
        
        /**
         * @brief Start the dashboard background update thread
         */
        void start();
        
        /**
         * @brief Stop the dashboard background update thread
         */
        void stop();
        
        /**
         * @brief Check if the dashboard is running
         * @return True if running
         */
        bool is_running() const;
        
        // Dashboard layout management
        Result<bool> create_dashboard_layout(const DashboardLayout& layout);
        Result<bool> update_dashboard_layout(const DashboardLayout& layout);
        Result<bool> delete_dashboard_layout(const std::string& layout_id);
        Result<DashboardLayout> get_dashboard_layout(const std::string& layout_id) const;
        Result<std::vector<DashboardLayout>> get_all_dashboard_layouts() const;
        Result<DashboardLayout> get_default_dashboard_layout() const;
        
        // Widget management
        Result<bool> add_widget_to_layout(const std::string& layout_id, const WidgetConfig& widget);
        Result<bool> remove_widget_from_layout(const std::string& layout_id, const std::string& widget_id);
        Result<bool> update_widget_in_layout(const std::string& layout_id, const WidgetConfig& widget);
        
        // Real-time metric data
        Result<std::vector<MetricDataPoint>> get_real_time_metrics(
            const std::string& query,
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time,
            int limit = 1000) const;
        
        Result<std::string> subscribe_to_metric_updates(
            const std::string& query,
            DashboardUpdateCallback callback);
        Result<bool> unsubscribe_from_metric_updates(const std::string& subscription_id);
        
        // Query pattern analysis
        Result<std::vector<QueryPattern>> analyze_query_patterns(
            const std::string& database_id,
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time) const;
        
        Result<std::vector<QueryPattern>> get_frequent_queries(
            const std::string& database_id = "",
            int limit = 10,
            std::chrono::hours time_window = std::chrono::hours(24)) const;
        
        Result<std::vector<QueryPattern>> get_slowest_queries(
            const std::string& database_id = "",
            int limit = 10,
            std::chrono::hours time_window = std::chrono::hours(24)) const;
        
        // Resource utilization
        Result<std::vector<ResourceUtilization>> get_current_resource_utilization(
            const std::string& resource_type = "") const;
        
        Result<std::vector<std::vector<double>>> get_resource_utilization_heatmap(
            const std::string& resource_type,
            int time_granularity_minutes,
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time) const;
        
        Result<std::vector<ResourceUtilization>> get_resource_recommendations() const;
        
        // Anomaly detection and alerts
        Result<std::vector<Anomaly>> detect_anomalies(
            const std::string& metric_name,
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time) const;
        
        Result<std::vector<Anomaly>> get_active_anomalies() const;
        Result<bool> acknowledge_anomaly(const std::string& anomaly_id, const std::string& user);
        Result<bool> resolve_anomaly(const std::string& anomaly_id, const std::string& user);
        
        // Alert management
        Result<bool> configure_alert(const AlertConfig& config);
        Result<bool> delete_alert(const std::string& alert_id);
        Result<AlertConfig> get_alert_config(const std::string& alert_id) const;
        Result<std::vector<AlertConfig>> get_all_alert_configs() const;
        Result<std::vector<AlertConfig>> get_triggered_alerts(
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time) const;
        
        // Export functionality
        Result<std::string> export_dashboard_data(
            const std::string& format = "json") const;
        
        Result<std::string> export_query_patterns(
            const std::string& database_id,
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time,
            const std::string& format = "json") const;
        
        Result<std::string> export_resource_utilization(
            const std::string& resource_type,
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time,
            const std::string& format = "json") const;
        
        Result<std::string> export_anomalies(
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time,
            const std::string& format = "json") const;
        
    private:
        // Background update thread function
        void background_update_loop();
        
        // Internal helper methods
        void update_dashboard_widgets();
        void check_for_anomalies();
        void send_alert_notifications();
        void cleanup_old_data();
        
        // Layout validation
        bool validate_dashboard_layout(const DashboardLayout& layout) const;
        bool validate_widget_config(const WidgetConfig& widget) const;
        
        // Data processing helpers
        std::vector<MetricDataPoint> process_metric_data(
            const std::vector<MetricDataPoint>& raw_data) const;
        
        std::vector<QueryPattern> process_query_patterns(
            const std::vector<QueryPattern>& raw_patterns) const;
        
        std::vector<ResourceUtilization> process_resource_utilization(
            const std::vector<ResourceUtilization>& raw_utilization) const;
        
        // Alert evaluation
        void evaluate_alert_conditions();
        bool should_trigger_alert(const AlertConfig& config, double current_value) const;
        void trigger_alert_notification(const AlertConfig& config, double current_value);
    };

} // namespace analytics
} // namespace jadevectordb

#endif // JADEVECTORDB_ANALYTICS_DASHBOARD_H