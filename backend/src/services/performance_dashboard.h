#ifndef JADEVECTORDB_PERFORMANCE_DASHBOARD_H
#define JADEVECTORDB_PERFORMANCE_DASHBOARD_H

#include "analytics_dashboard.h"
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace jadevectordb {
namespace analytics {

    /**
     * @brief Performance Metrics Dashboard
     * 
     * Specialized dashboard for system performance metrics
     */
    class PerformanceMetricsDashboard {
    private:
        std::shared_ptr<AnalyticsDashboardService> dashboard_service_;
        std::shared_ptr<logging::Logger> logger_;
        
    public:
        explicit PerformanceMetricsDashboard(std::shared_ptr<AnalyticsDashboardService> dashboard_service);
        
        /**
         * @brief Get CPU utilization metrics
         * @param hours Hours of historical data
         * @return Time series data for CPU utilization
         */
        Result<TimeSeriesData> get_cpu_utilization_metrics(int hours = 24) const;
        
        /**
         * @brief Get memory usage metrics
         * @param hours Hours of historical data
         * @return Time series data for memory usage
         */
        Result<TimeSeriesData> get_memory_usage_metrics(int hours = 24) const;
        
        /**
         * @brief Get disk I/O metrics
         * @param hours Hours of historical data
         * @return Time series data for disk I/O
         */
        Result<TimeSeriesData> get_disk_io_metrics(int hours = 24) const;
        
        /**
         * @brief Get network throughput metrics
         * @param hours Hours of historical data
         * @return Time series data for network throughput
         */
        Result<TimeSeriesData> get_network_throughput_metrics(int hours = 24) const;
        
        /**
         * @brief Get query response time metrics
         * @param hours Hours of historical data
         * @return Time series data for query response times
         */
        Result<TimeSeriesData> get_query_response_time_metrics(int hours = 24) const;
        
        /**
         * @brief Get vector insert rate metrics
         * @param hours Hours of historical data
         * @return Time series data for vector insert rates
         */
        Result<TimeSeriesData> get_vector_insert_rate_metrics(int hours = 24) const;
        
        /**
         * @brief Get similarity search latency metrics
         * @param hours Hours of historical data
         * @return Time series data for similarity search latencies
         */
        Result<TimeSeriesData> get_similarity_search_latency_metrics(int hours = 24) const;
        
        /**
         * @brief Get all performance metrics dashboard data
         * @return JSON data for the entire performance dashboard
         */
        Result<nlohmann::json> get_dashboard_data() const;
        
        /**
         * @brief Create the performance dashboard layout
         * @return Result indicating success or failure
         */
        Result<void> create_dashboard_layout() const;
    };

    /**
     * @brief Query Pattern Analysis Dashboard
     * 
     * Specialized dashboard for analyzing query patterns and behaviors
     */
    class QueryPatternAnalysisDashboard {
    private:
        std::shared_ptr<AnalyticsDashboardService> dashboard_service_;
        std::shared_ptr<logging::Logger> logger_;
        
    public:
        explicit QueryPatternAnalysisDashboard(std::shared_ptr<AnalyticsDashboardService> dashboard_service);
        
        /**
         * @brief Get query frequency analysis
         * @param hours Hours of historical data
         * @return Analysis of query frequencies by type
         */
        Result<nlohmann::json> get_query_frequency_analysis(int hours = 24) const;
        
        /**
         * @brief Get query latency distribution
         * @param hours Hours of historical data
         * @return Distribution data for query latencies
         */
        Result<nlohmann::json> get_query_latency_distribution(int hours = 24) const;
        
        /**
         * @brief Get query success rate analysis
         * @param hours Hours of historical data
         * @return Analysis of query success rates
         */
        Result<nlohmann::json> get_query_success_rate_analysis(int hours = 24) const;
        
        /**
         * @brief Get most frequent queries
         * @param limit Maximum number of queries to return
         * @return List of most frequent queries
         */
        Result<nlohmann::json> get_most_frequent_queries(int limit = 10) const;
        
        /**
         * @brief Get slowest queries
         * @param limit Maximum number of queries to return
         * @return List of slowest queries
         */
        Result<nlohmann::json> get_slowest_queries(int limit = 10) const;
        
        /**
         * @brief Get query pattern trends
         * @param hours Hours of historical data
         * @return Trend analysis of query patterns
         */
        Result<nlohmann::json> get_query_pattern_trends(int hours = 24) const;
        
        /**
         * @brief Get correlation analysis between query types
         * @param hours Hours of historical data
         * @return Correlation matrix of query types
         */
        Result<nlohmann::json> get_query_type_correlation(int hours = 24) const;
        
        /**
         * @brief Get all query pattern analysis dashboard data
         * @return JSON data for the entire query pattern dashboard
         */
        Result<nlohmann::json> get_dashboard_data() const;
        
        /**
         * @brief Create the query pattern analysis dashboard layout
         * @return Result indicating success or failure
         */
        Result<void> create_dashboard_layout() const;
    };

    /**
     * @brief Resource Utilization Dashboard
     * 
     * Specialized dashboard for monitoring resource utilization with heatmaps
     */
    class ResourceUtilizationDashboard {
    private:
        std::shared_ptr<AnalyticsDashboardService> dashboard_service_;
        std::shared_ptr<logging::Logger> logger_;
        
    public:
        explicit ResourceUtilizationDashboard(std::shared_ptr<AnalyticsDashboardService> dashboard_service);
        
        /**
         * @brief Get CPU utilization heatmap
         * @param hours Hours of historical data
         * @return Heatmap data for CPU utilization
         */
        Result<HeatmapData> get_cpu_utilization_heatmap(int hours = 24) const;
        
        /**
         * @brief Get memory usage heatmap
         * @param hours Hours of historical data
         * @return Heatmap data for memory usage
         */
        Result<HeatmapData> get_memory_usage_heatmap(int hours = 24) const;
        
        /**
         * @brief Get disk usage heatmap
         * @param hours Hours of historical data
         * @return Heatmap data for disk usage
         */
        Result<HeatmapData> get_disk_usage_heatmap(int hours = 24) const;
        
        /**
         * @brief Get network utilization heatmap
         * @param hours Hours of historical data
         * @return Heatmap data for network utilization
         */
        Result<HeatmapData> get_network_utilization_heatmap(int hours = 24) const;
        
        /**
         * @brief Get storage utilization by database
         * @return Storage utilization data by database
         */
        Result<nlohmann::json> get_storage_utilization_by_database() const;
        
        /**
         * @brief Get resource allocation efficiency
         * @return Efficiency metrics for resource allocation
         */
        Result<nlohmann::json> get_resource_allocation_efficiency() const;
        
        /**
         * @brief Get peak resource usage times
         * @param hours Hours of historical data
         * @return Analysis of peak resource usage times
         */
        Result<nlohmann::json> get_peak_resource_usage_times(int hours = 24) const;
        
        /**
         * @brief Get all resource utilization dashboard data
         * @return JSON data for the entire resource utilization dashboard
         */
        Result<nlohmann::json> get_dashboard_data() const;
        
        /**
         * @brief Create the resource utilization dashboard layout
         * @return Result indicating success or failure
         */
        Result<void> create_dashboard_layout() const;
    };

    /**
     * @brief Anomaly Detection Dashboard
     * 
     * Specialized dashboard for detecting and analyzing anomalies
     */
    class AnomalyDetectionDashboard {
    private:
        std::shared_ptr<AnalyticsDashboardService> dashboard_service_;
        std::shared_ptr<logging::Logger> logger_;
        std::vector<AlertConfig> anomaly_alerts_;
        
    public:
        explicit AnomalyDetectionDashboard(std::shared_ptr<AnalyticsDashboardService> dashboard_service);
        
        /**
         * @brief Detect performance anomalies
         * @param hours Hours of historical data to analyze
         * @return List of detected performance anomalies
         */
        Result<std::vector<AlertEvent>> detect_performance_anomalies(int hours = 24) const;
        
        /**
         * @brief Detect resource usage anomalies
         * @param hours Hours of historical data to analyze
         * @return List of detected resource usage anomalies
         */
        Result<std::vector<AlertEvent>> detect_resource_anomalies(int hours = 24) const;
        
        /**
         * @brief Detect query pattern anomalies
         * @param hours Hours of historical data to analyze
         * @return List of detected query pattern anomalies
         */
        Result<std::vector<AlertEvent>> detect_query_pattern_anomalies(int hours = 24) const;
        
        /**
         * @brief Get anomaly detection rules
         * @return Current anomaly detection rules
         */
        Result<std::vector<AlertConfig>> get_anomaly_detection_rules() const;
        
        /**
         * @brief Configure anomaly detection rule
         * @param rule Alert configuration for anomaly detection
         * @return Result indicating success or failure
         */
        Result<void> configure_anomaly_detection_rule(const AlertConfig& rule);
        
        /**
         * @brief Get recent anomalies
         * @param limit Maximum number of anomalies to return
         * @return List of recent anomalies
         */
        Result<std::vector<AlertEvent>> get_recent_anomalies(int limit = 50) const;
        
        /**
         * @brief Get anomaly trends
         * @param hours Hours of historical data
         * @return Trend analysis of anomalies
         */
        Result<nlohmann::json> get_anomaly_trends(int hours = 24) const;
        
        /**
         * @brief Get anomaly correlation analysis
         * @param hours Hours of historical data
         * @return Correlation analysis of different types of anomalies
         */
        Result<nlohmann::json> get_anomaly_correlation(int hours = 24) const;
        
        /**
         * @brief Get all anomaly detection dashboard data
         * @return JSON data for the entire anomaly detection dashboard
         */
        Result<nlohmann::json> get_dashboard_data() const;
        
        /**
         * @brief Create the anomaly detection dashboard layout
         * @return Result indicating success or failure
         */
        Result<void> create_dashboard_layout() const;
    };

} // namespace analytics
} // namespace jadevectordb

#endif // JADEVECTORDB_PERFORMANCE_DASHBOARD_H