#ifndef JADEVECTORDB_ANALYTICS_DASHBOARD_H
#define JADEVECTORDB_ANALYTICS_DASHBOARD_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <atomic>

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
        ChartType type;
        std::string metric_name;
        std::unordered_map<std::string, std::string> filters;
        int refresh_interval_seconds;
        int width;  // Grid width (1-12)
        int height; // Grid height (1-12)
        std::string color_scheme;
        bool show_legend;
        std::vector<std::string> dimensions; // For multi-dimensional data
        
        WidgetConfig() : refresh_interval_seconds(30), width(4, height(3), show_legend(true) {}
    };

    // Structure for dashboard layout
    struct DashboardLayout {
        std::string name;
        std::string description;
        std::vector<WidgetConfig> widgets;
        std::string theme; // "light", "dark", "auto"
        bool auto_refresh;
        int auto_refresh_interval_seconds;
        
        DashboardLayout() : theme("auto"), auto_refresh(true), auto_refresh_interval_seconds(30) {}
    };

    // Structure for time series data
    struct TimeSeriesData {
        std::string metric_name;
        std::vector<std::chrono::system_clock::time_point> timestamps;
        std::vector<double> values;
        std::unordered_map<std::string, std::string> labels;
        std::chrono::system_clock::time_point last_updated;
    };

    // Structure for heatmap data
    struct HeatmapData {
        std::string title;
        std::vector<std::string> x_labels;
        std::vector<std::string> y_labels;
        std::vector<std::vector<double>> values;
        std::chrono::system_clock::time_point last_updated;
    };

    // Structure for alert configuration
    struct AlertConfig {
        std::string id;
        std::string metric_name;
        std::string condition; // e.g., ">", "<", "==", ">="
        double threshold;
        std::string severity; // "info", "warning", "critical"
        std::string notification_channel; // "email", "slack", "webhook"
        std::string webhook_url;
        bool enabled;
        std::chrono::system_clock::time_point created_at;
        std::chrono::system_clock::time_point last_triggered;
        
        AlertConfig() : condition(">"), threshold(0.0), severity("warning"), enabled(true) {}
    };

    // Structure for alert event
    struct AlertEvent {
        std::string id;
        std::string alert_id;
        std::string metric_name;
        double current_value;
        double threshold;
        std::string severity;
        std::string message;
        std::chrono::system_clock::time_point triggered_at;
        bool acknowledged;
        std::chrono::system_clock::time_point acknowledged_at;
        std::string acknowledged_by;
    };

    /**
     * @brief Interface for data providers
     * 
     * This interface defines the contract for data sources that feed the dashboard
     */
    class IDataProvider {
    public:
        virtual ~IDataProvider() = default;
        
        /**
         * @brief Get time series data for a metric
         * @param metric_name Name of the metric
         * @param start_time Start time for data
         * @param end_time End time for data
         * @param granularity Data granularity (e.g., 1m, 5m, 1h)
         * @return Time series data
         */
        virtual Result<TimeSeriesData> get_time_series_data(
            const std::string& metric_name,
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time,
            const std::string& granularity = "1m") = 0;
        
        /**
         * @brief Get current metric value
         * @param metric_name Name of the metric
         * @return Current metric value
         */
        virtual Result<double> get_current_metric_value(const std::string& metric_name) = 0;
        
        /**
         * @brief Get heatmap data
         * @param data_type Type of heatmap data
         * @return Heatmap data
         */
        virtual Result<HeatmapData> get_heatmap_data(const std::string& data_type) = 0;
        
        /**
         * @brief Get available metrics
         * @return List of available metric names
         */
        virtual Result<std::vector<std::string>> get_available_metrics() = 0;
    };

    /**
     * @brief Metrics data provider implementation
     * 
     * This class provides data from the metrics service
     */
    class MetricsDataProvider : public IDataProvider {
    private:
        std::shared_ptr<MetricsService> metrics_service_;
        std::shared_ptr<MonitoringService> monitoring_service_;
        std::shared_ptr<logging::Logger> logger_;
        
    public:
        explicit MetricsDataProvider(std::shared_ptr<MetricsService> metrics_service,
                                   std::shared_ptr<MonitoringService> monitoring_service = nullptr);
        
        Result<TimeSeriesData> get_time_series_data(
            const std::string& metric_name,
            const std::chrono::system_clock::time_point& start_time,
            const std::chrono::system_clock::time_point& end_time,
            const std::string& granularity = "1m") override;
        
        Result<double> get_current_metric_value(const std::string& metric_name) override;
        
        Result<HeatmapData> get_heatmap_data(const std::string& data_type) override;
        
        Result<std::vector<std::string>> get_available_metrics() override;
        
    private:
        // Helper methods
        std::vector<double> aggregate_data(const std::vector<Metric>& metrics,
                                         const std::string& aggregation_type) const;
    };

    /**
     * @brief Analytics Dashboard Service
     * 
     * This service provides the core functionality for the analytics dashboard
     */
    class AnalyticsDashboardService {
    private:
        std::shared_ptr<logging::Logger> logger_;
        std::shared_ptr<IDataProvider> data_provider_;
        std::vector<DashboardLayout> dashboard_layouts_;
        std::unordered_map<std::string, AlertConfig> alert_configs_;
        std::vector<AlertEvent> alert_events_;
        std::mutex dashboard_mutex_;
        std::atomic<bool> running_;
        std::thread refresh_thread_;
        
    public:
        explicit AnalyticsDashboardService(std::shared_ptr<IDataProvider> data_provider);
        ~AnalyticsDashboardService();
        
        /**
         * @brief Initialize the dashboard service
         * @return True if initialization was successful
         */
        Result<void> initialize();
        
        /**
         * @brief Create a new dashboard layout
         * @param layout Dashboard layout configuration
         * @return True if creation was successful
         */
        Result<void> create_dashboard_layout(const DashboardLayout& layout);
        
        /**
         * @brief Get a dashboard layout by name
         * @param name Name of the layout
         * @return Dashboard layout
         */
        Result<DashboardLayout> get_dashboard_layout(const std::string& name) const;
        
        /**
         * @brief Get all dashboard layouts
         * @return List of all dashboard layouts
         */
        Result<std::vector<DashboardLayout>> get_all_dashboard_layouts() const;
        
        /**
         * @brief Update an existing dashboard layout
         * @param layout Updated dashboard layout
         * @return True if update was successful
         */
        Result<void> update_dashboard_layout(const DashboardLayout& layout);
        
        /**
         * @brief Delete a dashboard layout
         * @param name Name of the layout to delete
         * @return True if deletion was successful
         */
        Result<void> delete_dashboard_layout(const std::string& name);
        
        /**
         * @brief Get data for a specific widget
         * @param widget_config Configuration for the widget
         * @return Widget data in appropriate format
         */
        Result<nlohmann::json> get_widget_data(const WidgetConfig& widget_config) const;
        
        /**
         * @brief Get time series data for a metric
         * @param metric_name Name of the metric
         * @param hours Hours of historical data to retrieve
         * @return Time series data
         */
        Result<TimeSeriesData> get_metric_time_series(const std::string& metric_name, 
                                                   int hours = 24) const;
        
        /**
         * @brief Get current system health status
         * @return System health information
         */
        Result<SystemHealth> get_system_health() const;
        
        /**
         * @brief Get current database statuses
         * @return List of database statuses
         */
        Result<std::vector<DatabaseStatus>> get_database_statuses() const;
        
        /**
         * @brief Configure an alert
         * @param alert_config Alert configuration
         * @return True if configuration was successful
         */
        Result<void> configure_alert(const AlertConfig& alert_config);
        
        /**
         * @brief Get all alert configurations
         * @return List of alert configurations
         */
        Result<std::vector<AlertConfig>> get_alert_configurations() const;
        
        /**
         * @brief Get recent alert events
         * @param limit Maximum number of events to return
         * @return List of recent alert events
         */
        Result<std::vector<AlertEvent>> get_recent_alert_events(int limit = 50) const;
        
        /**
         * @brief Acknowledge an alert event
         * @param alert_event_id ID of the alert event to acknowledge
         * @param user User acknowledging the alert
         * @return True if acknowledgment was successful
         */
        Result<void> acknowledge_alert_event(const std::string& alert_event_id, 
                                          const std::string& user);
        
        /**
         * @brief Export dashboard data
         * @param format Export format ("json", "csv", "pdf")
         * @param dashboard_name Name of dashboard to export
         * @return Exported data
         */
        Result<std::string> export_dashboard_data(const std::string& format,
                                                const std::string& dashboard_name = "") const;
        
        /**
         * @brief Start background refresh thread
         */
        void start_background_refresh();
        
        /**
         * @brief Stop background refresh thread
         */
        void stop_background_refresh();
        
    private:
        // Internal helper methods
        
        /**
         * @brief Background thread for refreshing dashboard data
         */
        void refresh_loop();
        
        /**
         * @brief Check for triggered alerts
         */
        void check_alerts();
        
        /**
         * @brief Generate heatmap data for resource utilization
         * @return Heatmap data
         */
        HeatmapData generate_resource_heatmap() const;
        
        /**
         * @brief Generate query pattern analysis data
         * @return JSON data for query patterns
         */
        nlohmann::json generate_query_pattern_data() const;
        
        /**
         * @brief Format data for charting
         * @param data Time series data
         * @param chart_type Type of chart
         * @return Formatted data for charting library
         */
        nlohmann::json format_chart_data(const TimeSeriesData& data, ChartType chart_type) const;
        
        /**
         * @brief Validate dashboard layout
         * @param layout Layout to validate
         * @return True if valid
         */
        bool validate_layout(const DashboardLayout& layout) const;
        
        /**
         * @brief Generate unique ID
         * @return Unique ID string
         */
        std::string generate_unique_id() const;
    };

    // Forward declarations for specific dashboard components
    class PerformanceMetricsDashboard;
    class QueryPatternAnalysisDashboard;
    class ResourceUtilizationDashboard;
    class AnomalyDetectionDashboard;

} // namespace analytics
} // namespace jadevectordb

#endif // JADEVECTORDB_ANALYTICS_DASHBOARD_H