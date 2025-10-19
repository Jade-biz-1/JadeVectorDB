#include "analytics_dashboard.h"
#include <nlohmann/json.hpp>
#include <random>
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace jadevectordb {
namespace analytics {

// MetricsDataProvider implementation
MetricsDataProvider::MetricsDataProvider(std::shared_ptr<MetricsService> metrics_service,
                                       std::shared_ptr<MonitoringService> monitoring_service)
    : metrics_service_(metrics_service), monitoring_service_(monitoring_service) {
    logger_ = logging::LoggerManager::get_logger("MetricsDataProvider");
}

Result<TimeSeriesData> MetricsDataProvider::get_time_series_data(
    const std::string& metric_name,
    const std::chrono::system_clock::time_point& start_time,
    const std::chrono::system_clock::time_point& end_time,
    const std::string& granularity) {
    
    if (!metrics_service_) {
        RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Metrics service not available");
    }
    
    TimeSeriesData data;
    data.metric_name = metric_name;
    data.last_updated = std::chrono::system_clock::now();
    
    // In a real implementation, we would retrieve historical data from the metrics service
    // For this implementation, we'll generate sample data for demonstration
    
    auto duration = std::chrono::duration_cast<std::chrono::hours>(end_time - start_time).count();
    int points = static_cast<int>(duration);
    
    data.timestamps.reserve(points);
    data.values.reserve(points);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);
    
    for (int i = 0; i < points; ++i) {
        auto timestamp = start_time + std::chrono::hours(i);
        data.timestamps.push_back(timestamp);
        data.values.push_back(dis(gen));
    }
    
    return Result<TimeSeriesData>::success(data);
}

Result<double> MetricsDataProvider::get_current_metric_value(const std::string& metric_name) {
    if (!metrics_service_) {
        RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Metrics service not available");
    }
    
    // In a real implementation, we would get the current value from the metrics service
    // For this implementation, we'll return a random value for demonstration
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);
    
    return Result<double>::success(dis(gen));
}

Result<HeatmapData> MetricsDataProvider::get_heatmap_data(const std::string& data_type) {
    HeatmapData data;
    data.title = data_type;
    data.last_updated = std::chrono::system_clock::now();
    
    // Generate sample heatmap data
    data.x_labels = {"00:00", "04:00", "08:00", "12:00", "16:00", "20:00"};
    data.y_labels = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
    
    data.values.resize(data.y_labels.size());
    for (auto& row : data.values) {
        row.resize(data.x_labels.size());
        for (auto& cell : row) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 100.0);
            cell = dis(gen);
        }
    }
    
    return Result<HeatmapData>::success(data);
}

Result<std::vector<std::string>> MetricsDataProvider::get_available_metrics() {
    // In a real implementation, we would get this from the metrics service
    std::vector<std::string> metrics = {
        "cpu_utilization",
        "memory_usage",
        "disk_io",
        "network_throughput",
        "query_response_time",
        "vector_insert_rate",
        "similarity_search_latency",
        "index_build_time",
        "storage_usage",
        "concurrent_connections"
    };
    
    return Result<std::vector<std::string>>::success(metrics);
}

// AnalyticsDashboardService implementation
AnalyticsDashboardService::AnalyticsDashboardService(std::shared_ptr<IDataProvider> data_provider)
    : data_provider_(data_provider), running_(false) {
    logger_ = logging::LoggerManager::get_logger("AnalyticsDashboardService");
}

AnalyticsDashboardService::~AnalyticsDashboardService() {
    stop_background_refresh();
}

Result<void> AnalyticsDashboardService::initialize() {
    if (!data_provider_) {
        RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Data provider not configured");
    }
    
    // Create default dashboards
    DashboardLayout perf_layout;
    perf_layout.name = "performance";
    perf_layout.description = "Performance Metrics Dashboard";
    perf_layout.theme = "dark";
    
    WidgetConfig cpu_widget;
    cpu_widget.id = "cpu_utilization";
    cpu_widget.title = "CPU Utilization";
    cpu_widget.description = "Current CPU usage percentage";
    cpu_widget.type = ChartType::LINE_CHART;
    cpu_widget.metric_name = "cpu_utilization";
    cpu_widget.refresh_interval_seconds = 30;
    cpu_widget.width = 6;
    cpu_widget.height = 4;
    perf_layout.widgets.push_back(cpu_widget);
    
    WidgetConfig memory_widget;
    memory_widget.id = "memory_usage";
    memory_widget.title = "Memory Usage";
    memory_widget.description = "Current memory usage percentage";
    memory_widget.type = ChartType::LINE_CHART;
    memory_widget.metric_name = "memory_usage";
    memory_widget.refresh_interval_seconds = 30;
    memory_widget.width = 6;
    memory_widget.height = 4;
    perf_layout.widgets.push_back(memory_widget);
    
    {
        std::lock_guard<std::mutex> lock(dashboard_mutex_);
        dashboard_layouts_.push_back(perf_layout);
    }
    
    LOG_INFO(logger_, "Analytics dashboard service initialized");
    return Result<void>::success();
}

Result<void> AnalyticsDashboardService::create_dashboard_layout(const DashboardLayout& layout) {
    if (!validate_layout(layout)) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid dashboard layout");
    }
    
    {
        std::lock_guard<std::mutex> lock(dashboard_mutex_);
        // Check if layout with same name already exists
        for (const auto& existing_layout : dashboard_layouts_) {
            if (existing_layout.name == layout.name) {
                RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "Dashboard layout already exists: " + layout.name);
            }
        }
        dashboard_layouts_.push_back(layout);
    }
    
    LOG_INFO(logger_, "Created dashboard layout: " + layout.name);
    return Result<void>::success();
}

Result<DashboardLayout> AnalyticsDashboardService::get_dashboard_layout(const std::string& name) const {
    std::lock_guard<std::mutex> lock(dashboard_mutex_);
    
    for (const auto& layout : dashboard_layouts_) {
        if (layout.name == name) {
            return Result<DashboardLayout>::success(layout);
        }
    }
    
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Dashboard layout not found: " + name);
}

Result<std::vector<DashboardLayout>> AnalyticsDashboardService::get_all_dashboard_layouts() const {
    std::lock_guard<std::mutex> lock(dashboard_mutex_);
    return Result<std::vector<DashboardLayout>>::success(dashboard_layouts_);
}

Result<void> AnalyticsDashboardService::update_dashboard_layout(const DashboardLayout& layout) {
    if (!validate_layout(layout)) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid dashboard layout");
    }
    
    {
        std::lock_guard<std::mutex> lock(dashboard_mutex_);
        for (auto& existing_layout : dashboard_layouts_) {
            if (existing_layout.name == layout.name) {
                existing_layout = layout;
                LOG_INFO(logger_, "Updated dashboard layout: " + layout.name);
                return Result<void>::success();
            }
        }
    }
    
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Dashboard layout not found: " + layout.name);
}

Result<void> AnalyticsDashboardService::delete_dashboard_layout(const std::string& name) {
    std::lock_guard<std::mutex> lock(dashboard_mutex_);
    
    for (auto it = dashboard_layouts_.begin(); it != dashboard_layouts_.end(); ++it) {
        if (it->name == name) {
            dashboard_layouts_.erase(it);
            LOG_INFO(logger_, "Deleted dashboard layout: " + name);
            return Result<void>::success();
        }
    }
    
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Dashboard layout not found: " + name);
}

Result<nlohmann::json> AnalyticsDashboardService::get_widget_data(const WidgetConfig& widget_config) const {
    if (!data_provider_) {
        RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Data provider not available");
    }
    
    nlohmann::json widget_data;
    
    try {
        switch (widget_config.type) {
            case ChartType::LINE_CHART:
            case ChartType::BAR_CHART: {
                // Get time series data
                auto now = std::chrono::system_clock::now();
                auto start_time = now - std::chrono::hours(24);
                
                auto ts_result = data_provider_->get_time_series_data(
                    widget_config.metric_name, start_time, now);
                
                if (ts_result.has_value()) {
                    widget_data = format_chart_data(ts_result.value(), widget_config.type);
                } else {
                    widget_data["error"] = "Failed to retrieve time series data";
                }
                break;
            }
            
            case ChartType::GAUGE: {
                // Get current value
                auto value_result = data_provider_->get_current_metric_value(widget_config.metric_name);
                if (value_result.has_value()) {
                    widget_data["value"] = value_result.value();
                    widget_data["title"] = widget_config.title;
                    widget_data["description"] = widget_config.description;
                } else {
                    widget_data["error"] = "Failed to retrieve current value";
                }
                break;
            }
            
            case ChartType::HEATMAP: {
                // Get heatmap data
                auto heatmap_result = data_provider_->get_heatmap_data(widget_config.metric_name);
                if (heatmap_result.has_value()) {
                    const auto& heatmap_data = heatmap_result.value();
                    widget_data["title"] = heatmap_data.title;
                    widget_data["x_labels"] = heatmap_data.x_labels;
                    widget_data["y_labels"] = heatmap_data.y_labels;
                    widget_data["values"] = heatmap_data.values;
                } else {
                    widget_data["error"] = "Failed to retrieve heatmap data";
                }
                break;
            }
            
            default: {
                widget_data["error"] = "Unsupported chart type";
                break;
            }
        }
    } catch (const std::exception& e) {
        widget_data["error"] = std::string("Exception: ") + e.what();
    }
    
    return Result<nlohmann::json>::success(widget_data);
}

Result<TimeSeriesData> AnalyticsDashboardService::get_metric_time_series(const std::string& metric_name, 
                                                                       int hours) const {
    if (!data_provider_) {
        RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Data provider not available");
    }
    
    auto end_time = std::chrono::system_clock::now();
    auto start_time = end_time - std::chrono::hours(hours);
    
    return data_provider_->get_time_series_data(metric_name, start_time, end_time);
}

Result<SystemHealth> AnalyticsDashboardService::get_system_health() const {
    // In a real implementation, we would get this from the monitoring service
    SystemHealth health;
    health.status = "healthy";
    health.checked_at = std::chrono::system_clock::now();
    health.components["database"] = "healthy";
    health.components["indexing"] = "healthy";
    health.components["search"] = "healthy";
    health.metrics["cpu_utilization"] = 45.2f;
    health.metrics["memory_usage"] = 67.8f;
    health.metrics["disk_usage"] = 34.1f;
    health.uptime_seconds = 86400; // 24 hours
    
    return Result<SystemHealth>::success(health);
}

Result<std::vector<DatabaseStatus>> AnalyticsDashboardService::get_database_statuses() const {
    // In a real implementation, we would get this from the monitoring service
    std::vector<DatabaseStatus> statuses;
    
    DatabaseStatus status1;
    status1.database_id = "db1";
    status1.status = "online";
    status1.vector_count = 1000000;
    status1.index_count = 5;
    status1.storage_used_bytes = 500000000; // 500MB
    status1.query_performance_ms = 15.2f;
    status1.storage_utilization = 45.0f;
    status1.uptime_seconds = 86400;
    statuses.push_back(status1);
    
    DatabaseStatus status2;
    status2.database_id = "db2";
    status2.status = "online";
    status2.vector_count = 500000;
    status2.index_count = 3;
    status2.storage_used_bytes = 250000000; // 250MB
    status2.query_performance_ms = 12.8f;
    status2.storage_utilization = 25.0f;
    status2.uptime_seconds = 43200;
    statuses.push_back(status2);
    
    return Result<std::vector<DatabaseStatus>>::success(statuses);
}

Result<void> AnalyticsDashboardService::configure_alert(const AlertConfig& alert_config) {
    std::lock_guard<std::mutex> lock(dashboard_mutex_);
    
    if (alert_config.id.empty()) {
        AlertConfig config = alert_config;
        config.id = generate_unique_id();
        config.created_at = std::chrono::system_clock::now();
        alert_configs_[config.id] = config;
    } else {
        alert_configs_[alert_config.id] = alert_config;
    }
    
    LOG_INFO(logger_, "Configured alert: " + alert_config.id);
    return Result<void>::success();
}

Result<std::vector<AlertConfig>> AnalyticsDashboardService::get_alert_configurations() const {
    std::lock_guard<std::mutex> lock(dashboard_mutex_);
    
    std::vector<AlertConfig> configs;
    configs.reserve(alert_configs_.size());
    
    for (const auto& pair : alert_configs_) {
        configs.push_back(pair.second);
    }
    
    return Result<std::vector<AlertConfig>>::success(configs);
}

Result<std::vector<AlertEvent>> AnalyticsDashboardService::get_recent_alert_events(int limit) const {
    std::lock_guard<std::mutex> lock(dashboard_mutex_);
    
    std::vector<AlertEvent> events;
    size_t start_index = 0;
    
    if (alert_events_.size() > static_cast<size_t>(limit)) {
        start_index = alert_events_.size() - limit;
    }
    
    events.reserve(alert_events_.size() - start_index);
    for (size_t i = start_index; i < alert_events_.size(); ++i) {
        events.push_back(alert_events_[i]);
    }
    
    return Result<std::vector<AlertEvent>>::success(events);
}

Result<void> AnalyticsDashboardService::acknowledge_alert_event(const std::string& alert_event_id, 
                                                             const std::string& user) {
    std::lock_guard<std::mutex> lock(dashboard_mutex_);
    
    for (auto& event : alert_events_) {
        if (event.id == alert_event_id && !event.acknowledged) {
            event.acknowledged = true;
            event.acknowledged_at = std::chrono::system_clock::now();
            event.acknowledged_by = user;
            LOG_INFO(logger_, "Acknowledged alert event: " + alert_event_id);
            return Result<void>::success();
        }
    }
    
    RETURN_ERROR(ErrorCode::NOT_FOUND, "Alert event not found or already acknowledged: " + alert_event_id);
}

Result<std::string> AnalyticsDashboardService::export_dashboard_data(const std::string& format,
                                                                  const std::string& dashboard_name) const {
    nlohmann::json export_data;
    
    // Add system health
    auto health_result = get_system_health();
    if (health_result.has_value()) {
        const auto& health = health_result.value();
        export_data["system_health"]["status"] = health.status;
        export_data["system_health"]["uptime_seconds"] = health.uptime_seconds;
        for (const auto& component : health.components) {
            export_data["system_health"]["components"][component.first] = component.second;
        }
        for (const auto& metric : health.metrics) {
            export_data["system_health"]["metrics"][metric.first] = metric.second;
        }
    }
    
    // Add database statuses
    auto db_result = get_database_statuses();
    if (db_result.has_value()) {
        for (const auto& db_status : db_result.value()) {
            export_data["databases"][db_status.database_id]["status"] = db_status.status;
            export_data["databases"][db_status.database_id]["vector_count"] = db_status.vector_count;
            export_data["databases"][db_status.database_id]["storage_used_mb"] = 
                static_cast<double>(db_status.storage_used_bytes) / (1024.0 * 1024.0);
            export_data["databases"][db_status.database_id]["query_performance_ms"] = 
                db_status.query_performance_ms;
        }
    }
    
    if (format == "json") {
        return Result<std::string>::success(export_data.dump(2));
    } else {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Unsupported export format: " + format);
    }
}

void AnalyticsDashboardService::start_background_refresh() {
    if (running_.exchange(true)) {
        return; // Already running
    }
    
    refresh_thread_ = std::thread(&AnalyticsDashboardService::refresh_loop, this);
    LOG_INFO(logger_, "Started background refresh thread");
}

void AnalyticsDashboardService::stop_background_refresh() {
    if (!running_.exchange(false)) {
        return; // Not running
    }
    
    if (refresh_thread_.joinable()) {
        refresh_thread_.join();
    }
    
    LOG_INFO(logger_, "Stopped background refresh thread");
}

void AnalyticsDashboardService::refresh_loop() {
    while (running_) {
        try {
            // Refresh dashboard data
            check_alerts();
            
            // Sleep for the configured interval
            std::this_thread::sleep_for(std::chrono::seconds(30));
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Error in refresh loop: " + std::string(e.what()));
        }
    }
}

void AnalyticsDashboardService::check_alerts() {
    // In a real implementation, we would check metrics against alert thresholds
    // For this implementation, we'll just log that alerts are being checked
    LOG_DEBUG(logger_, "Checking alerts...");
}

HeatmapData AnalyticsDashboardService::generate_resource_heatmap() const {
    HeatmapData data;
    data.title = "Resource Utilization Heatmap";
    data.last_updated = std::chrono::system_clock::now();
    
    // Generate sample data
    data.x_labels = {"CPU", "Memory", "Disk", "Network"};
    data.y_labels = {"Server 1", "Server 2", "Server 3", "Server 4", "Server 5"};
    
    data.values.resize(data.y_labels.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);
    
    for (auto& row : data.values) {
        row.resize(data.x_labels.size());
        for (auto& cell : row) {
            cell = dis(gen);
        }
    }
    
    return data;
}

nlohmann::json AnalyticsDashboardService::generate_query_pattern_data() const {
    nlohmann::json data;
    
    // Generate sample query pattern data
    data["patterns"] = nlohmann::json::array();
    
    nlohmann::json pattern1;
    pattern1["name"] = "Similarity Search";
    pattern1["frequency"] = 45;
    pattern1["average_latency_ms"] = 15.2;
    pattern1["success_rate"] = 99.8;
    data["patterns"].push_back(pattern1);
    
    nlohmann::json pattern2;
    pattern2["name"] = "Batch Insert";
    pattern2["frequency"] = 12;
    pattern2["average_latency_ms"] = 234.7;
    pattern2["success_rate"] = 98.5;
    data["patterns"].push_back(pattern2);
    
    nlohmann::json pattern3;
    pattern3["name"] = "Metadata Filter";
    pattern3["frequency"] = 33;
    pattern3["average_latency_ms"] = 8.4;
    pattern3["success_rate"] = 99.9;
    data["patterns"].push_back(pattern3);
    
    return data;
}

nlohmann::json AnalyticsDashboardService::format_chart_data(const TimeSeriesData& data, ChartType chart_type) const {
    nlohmann::json chart_data;
    
    chart_data["labels"] = nlohmann::json::array();
    chart_data["datasets"] = nlohmann::json::array();
    
    nlohmann::json dataset;
    dataset["label"] = data.metric_name;
    dataset["data"] = nlohmann::json::array();
    
    for (size_t i = 0; i < data.timestamps.size() && i < data.values.size(); ++i) {
        // Format timestamp
        auto time_t = std::chrono::system_clock::to_time_t(data.timestamps[i]);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%H:%M");
        chart_data["labels"].push_back(ss.str());
        dataset["data"].push_back(data.values[i]);
    }
    
    chart_data["datasets"].push_back(dataset);
    
    return chart_data;
}

bool AnalyticsDashboardService::validate_layout(const DashboardLayout& layout) const {
    if (layout.name.empty()) {
        return false;
    }
    
    if (layout.widgets.empty()) {
        return false;
    }
    
    for (const auto& widget : layout.widgets) {
        if (widget.id.empty() || widget.title.empty() || widget.metric_name.empty()) {
            return false;
        }
        
        if (widget.width <= 0 || widget.width > 12 || widget.height <= 0 || widget.height > 12) {
            return false;
        }
    }
    
    return true;
}

std::string AnalyticsDashboardService::generate_unique_id() const {
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    
    return "dash_" + std::to_string(now) + "_" + std::to_string(dis(gen));
}

} // namespace analytics
} // namespace jadevectordb