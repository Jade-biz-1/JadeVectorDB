#include "performance_dashboard.h"
#include "analytics_dashboard.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <chrono>

using namespace jadevectordb::analytics;
using ::testing::Return;
using ::testing::_;

// Mock data provider for testing
class MockDataProvider : public IDataProvider {
public:
    MOCK_METHOD(Result<TimeSeriesData>, get_time_series_data, 
                (const std::string& metric_name,
                 const std::chrono::system_clock::time_point& start_time,
                 const std::chrono::system_clock::time_point& end_time,
                 const std::string& granularity), (override));
    
    MOCK_METHOD(Result<double>, get_current_metric_value, 
                (const std::string& metric_name), (override));
    
    MOCK_METHOD(Result<HeatmapData>, get_heatmap_data, 
                (const std::string& data_type), (override));
    
    MOCK_METHOD(Result<std::vector<std::string>>, get_available_metrics, (), (override));
};

// Test fixture for analytics dashboard
class AnalyticsDashboardTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_data_provider_ = std::make_shared<MockDataProvider>();
        dashboard_service_ = std::make_shared<AnalyticsDashboardService>(mock_data_provider_);
        
        auto init_result = dashboard_service_->initialize();
        ASSERT_TRUE(init_result.has_value());
    }
    
    void TearDown() override {
        dashboard_service_.reset();
        mock_data_provider_.reset();
    }
    
    std::shared_ptr<MockDataProvider> mock_data_provider_;
    std::shared_ptr<AnalyticsDashboardService> dashboard_service_;
};

// Test basic dashboard service initialization
TEST_F(AnalyticsDashboardTest, InitializeDashboardService) {
    // Service should already be initialized in SetUp
    EXPECT_NE(dashboard_service_, nullptr);
}

// Test creating dashboard layouts
TEST_F(AnalyticsDashboardTest, CreateDashboardLayout) {
    DashboardLayout layout;
    layout.name = "test_layout";
    layout.description = "Test layout for unit testing";
    layout.theme = "light";
    
    WidgetConfig widget;
    widget.id = "test_widget";
    widget.title = "Test Widget";
    widget.description = "A test widget";
    widget.type = ChartType::LINE_CHART;
    widget.metric_name = "test_metric";
    widget.refresh_interval_seconds = 30;
    widget.width = 6;
    widget.height = 4;
    layout.widgets.push_back(widget);
    
    auto result = dashboard_service_->create_dashboard_layout(layout);
    EXPECT_TRUE(result.has_value());
    
    // Retrieve the layout
    auto retrieved_result = dashboard_service_->get_dashboard_layout("test_layout");
    EXPECT_TRUE(retrieved_result.has_value());
    EXPECT_EQ(retrieved_result.value().name, "test_layout");
    EXPECT_EQ(retrieved_result.value().widgets.size(), 1);
}

// Test getting all dashboard layouts
TEST_F(AnalyticsDashboardTest, GetAllDashboardLayouts) {
    // Should have at least the default performance layout
    auto result = dashboard_service_->get_all_dashboard_layouts();
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(result.value().empty());
}

// Test updating dashboard layouts
TEST_F(AnalyticsDashboardTest, UpdateDashboardLayout) {
    // Create a layout first
    DashboardLayout layout;
    layout.name = "update_test_layout";
    layout.description = "Layout for update testing";
    layout.theme = "dark";
    
    WidgetConfig widget;
    widget.id = "update_test_widget";
    widget.title = "Update Test Widget";
    widget.description = "A widget for update testing";
    widget.type = ChartType::BAR_CHART;
    widget.metric_name = "update_test_metric";
    widget.refresh_interval_seconds = 60;
    widget.width = 8;
    widget.height = 5;
    layout.widgets.push_back(widget);
    
    auto create_result = dashboard_service_->create_dashboard_layout(layout);
    EXPECT_TRUE(create_result.has_value());
    
    // Update the layout
    layout.description = "Updated layout description";
    layout.theme = "light";
    
    auto update_result = dashboard_service_->update_dashboard_layout(layout);
    EXPECT_TRUE(update_result.has_value());
    
    // Verify the update
    auto retrieved_result = dashboard_service_->get_dashboard_layout("update_test_layout");
    EXPECT_TRUE(retrieved_result.has_value());
    EXPECT_EQ(retrieved_result.value().description, "Updated layout description");
    EXPECT_EQ(retrieved_result.value().theme, "light");
}

// Test deleting dashboard layouts
TEST_F(AnalyticsDashboardTest, DeleteDashboardLayout) {
    // Create a layout first
    DashboardLayout layout;
    layout.name = "delete_test_layout";
    layout.description = "Layout for deletion testing";
    layout.theme = "dark";
    
    WidgetConfig widget;
    widget.id = "delete_test_widget";
    widget.title = "Delete Test Widget";
    widget.description = "A widget for deletion testing";
    widget.type = ChartType::PIE_CHART;
    widget.metric_name = "delete_test_metric";
    widget.refresh_interval_seconds = 45;
    widget.width = 6;
    widget.height = 4;
    layout.widgets.push_back(widget);
    
    auto create_result = dashboard_service_->create_dashboard_layout(layout);
    EXPECT_TRUE(create_result.has_value());
    
    // Delete the layout
    auto delete_result = dashboard_service_->delete_dashboard_layout("delete_test_layout");
    EXPECT_TRUE(delete_result.has_value());
    
    // Verify deletion
    auto retrieved_result = dashboard_service_->get_dashboard_layout("delete_test_layout");
    EXPECT_FALSE(retrieved_result.has_value());
}

// Test performance metrics dashboard
TEST_F(AnalyticsDashboardTest, PerformanceMetricsDashboard) {
    auto perf_dashboard = std::make_shared<PerformanceMetricsDashboard>(dashboard_service_);
    
    // Test creating dashboard layout
    auto create_result = perf_dashboard->create_dashboard_layout();
    EXPECT_TRUE(create_result.has_value());
    
    // Test getting dashboard data
    auto data_result = perf_dashboard->get_dashboard_data();
    EXPECT_TRUE(data_result.has_value());
}

// Test query pattern analysis dashboard
TEST_F(AnalyticsDashboardTest, QueryPatternAnalysisDashboard) {
    auto query_dashboard = std::make_shared<QueryPatternAnalysisDashboard>(dashboard_service_);
    
    // Test creating dashboard layout
    auto create_result = query_dashboard->create_dashboard_layout();
    EXPECT_TRUE(create_result.has_value());
    
    // Test getting dashboard data
    auto data_result = query_dashboard->get_dashboard_data();
    EXPECT_TRUE(data_result.has_value());
}

// Test resource utilization dashboard
TEST_F(AnalyticsDashboardTest, ResourceUtilizationDashboard) {
    auto resource_dashboard = std::make_shared<ResourceUtilizationDashboard>(dashboard_service_);
    
    // Test creating dashboard layout
    auto create_result = resource_dashboard->create_dashboard_layout();
    EXPECT_TRUE(create_result.has_value());
    
    // Test getting dashboard data
    auto data_result = resource_dashboard->get_dashboard_data();
    EXPECT_TRUE(data_result.has_value());
}

// Test anomaly detection dashboard
TEST_F(AnalyticsDashboardTest, AnomalyDetectionDashboard) {
    auto anomaly_dashboard = std::make_shared<AnomalyDetectionDashboard>(dashboard_service_);
    
    // Test creating dashboard layout
    auto create_result = anomaly_dashboard->create_dashboard_layout();
    EXPECT_TRUE(create_result.has_value());
    
    // Test getting dashboard data
    auto data_result = anomaly_dashboard->get_dashboard_data();
    EXPECT_TRUE(data_result.has_value());
}

// Test system health monitoring
TEST_F(AnalyticsDashboardTest, SystemHealthMonitoring) {
    auto health_result = dashboard_service_->get_system_health();
    EXPECT_TRUE(health_result.has_value());
    
    const auto& health = health_result.value();
    EXPECT_FALSE(health.status.empty());
    EXPECT_FALSE(health.components.empty());
    EXPECT_FALSE(health.metrics.empty());
}

// Test database status monitoring
TEST_F(AnalyticsDashboardTest, DatabaseStatusMonitoring) {
    auto status_result = dashboard_service_->get_database_statuses();
    EXPECT_TRUE(status_result.has_value());
    
    const auto& statuses = status_result.value();
    EXPECT_FALSE(statuses.empty());
    
    for (const auto& status : statuses) {
        EXPECT_FALSE(status.database_id.empty());
        EXPECT_FALSE(status.status.empty());
        EXPECT_GE(status.vector_count, 0);
        EXPECT_GE(status.storage_used_bytes, 0);
    }
}

// Test alert configuration
TEST_F(AnalyticsDashboardTest, AlertConfiguration) {
    AlertConfig alert_config;
    alert_config.id = "test_alert";
    alert_config.metric_name = "cpu_utilization";
    alert_config.condition = ">";
    alert_config.threshold = 90.0;
    alert_config.severity = "critical";
    alert_config.enabled = true;
    
    auto config_result = dashboard_service_->configure_alert(alert_config);
    EXPECT_TRUE(config_result.has_value());
    
    // Test getting alert configurations
    auto get_result = dashboard_service_->get_alert_configurations();
    EXPECT_TRUE(get_result.has_value());
    EXPECT_FALSE(get_result.value().empty());
}

// Test data provider integration
TEST_F(AnalyticsDashboardTest, DataProviderIntegration) {
    TimeSeriesData mock_data;
    mock_data.metric_name = "test_metric";
    mock_data.last_updated = std::chrono::system_clock::now();
    mock_data.timestamps.push_back(std::chrono::system_clock::now());
    mock_data.values.push_back(42.0);
    
    // Set up mock expectations
    EXPECT_CALL(*mock_data_provider_, get_time_series_data(_, _, _, _))
        .WillOnce(Return(Result<TimeSeriesData>::success(mock_data)));
    
    EXPECT_CALL(*mock_data_provider_, get_current_metric_value(_))
        .WillOnce(Return(Result<double>::success(42.0)));
    
    HeatmapData mock_heatmap;
    mock_heatmap.title = "test_heatmap";
    mock_heatmap.last_updated = std::chrono::system_clock::now();
    mock_heatmap.x_labels = {"A", "B"};
    mock_heatmap.y_labels = {"X", "Y"};
    mock_heatmap.values = {{1.0, 2.0}, {3.0, 4.0}};
    
    EXPECT_CALL(*mock_data_provider_, get_heatmap_data(_))
        .WillOnce(Return(Result<HeatmapData>::success(mock_heatmap)));
    
    std::vector<std::string> mock_metrics = {"metric1", "metric2", "metric3"};
    EXPECT_CALL(*mock_data_provider_, get_available_metrics())
        .WillOnce(Return(Result<std::vector<std::string>>::success(mock_metrics)));
    
    // Test the data provider methods
    auto ts_result = mock_data_provider_->get_time_series_data(
        "test_metric", 
        std::chrono::system_clock::now() - std::chrono::hours(1),
        std::chrono::system_clock::now());
    EXPECT_TRUE(ts_result.has_value());
    EXPECT_EQ(ts_result.value().metric_name, "test_metric");
    
    auto current_result = mock_data_provider_->get_current_metric_value("test_metric");
    EXPECT_TRUE(current_result.has_value());
    EXPECT_DOUBLE_EQ(current_result.value(), 42.0);
    
    auto heatmap_result = mock_data_provider_->get_heatmap_data("test_heatmap");
    EXPECT_TRUE(heatmap_result.has_value());
    EXPECT_EQ(heatmap_result.value().title, "test_heatmap");
    
    auto metrics_result = mock_data_provider_->get_available_metrics();
    EXPECT_TRUE(metrics_result.has_value());
    EXPECT_EQ(metrics_result.value().size(), 3);
}

// Test dashboard data export
TEST_F(AnalyticsDashboardTest, DashboardDataExport) {
    auto export_result = dashboard_service_->export_dashboard_data("json");
    EXPECT_TRUE(export_result.has_value());
    EXPECT_FALSE(export_result.value().empty());
}

// Test background refresh functionality
TEST_F(AnalyticsDashboardTest, BackgroundRefresh) {
    // Start background refresh
    dashboard_service_->start_background_refresh();
    
    // Give it a moment to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Stop background refresh
    dashboard_service_->stop_background_refresh();
    
    SUCCEED();
}

// Test widget data retrieval
TEST_F(AnalyticsDashboardTest, WidgetDataRetrieval) {
    WidgetConfig widget_config;
    widget_config.id = "test_widget";
    widget_config.title = "Test Widget";
    widget_config.description = "A test widget";
    widget_config.type = ChartType::LINE_CHART;
    widget_config.metric_name = "test_metric";
    widget_config.refresh_interval_seconds = 30;
    widget_config.width = 6;
    widget_config.height = 4;
    
    TimeSeriesData mock_data;
    mock_data.metric_name = "test_metric";
    mock_data.last_updated = std::chrono::system_clock::now();
    mock_data.timestamps.push_back(std::chrono::system_clock::now());
    mock_data.values.push_back(42.0);
    
    // Set up mock expectations
    EXPECT_CALL(*mock_data_provider_, get_time_series_data(_, _, _, _))
        .WillOnce(Return(Result<TimeSeriesData>::success(mock_data)));
    
    auto data_result = dashboard_service_->get_widget_data(widget_config);
    EXPECT_TRUE(data_result.has_value());
}

// Test metric time series retrieval
TEST_F(AnalyticsDashboardTest, MetricTimeSeriesRetrieval) {
    TimeSeriesData mock_data;
    mock_data.metric_name = "cpu_utilization";
    mock_data.last_updated = std::chrono::system_clock::now();
    
    auto now = std::chrono::system_clock::now();
    for (int i = 0; i < 24; ++i) {
        mock_data.timestamps.push_back(now - std::chrono::hours(24 - i));
        mock_data.values.push_back(50.0 + i); // Increasing values for testing
    }
    
    // Set up mock expectations
    EXPECT_CALL(*mock_data_provider_, get_time_series_data(_, _, _, _))
        .WillOnce(Return(Result<TimeSeriesData>::success(mock_data)));
    
    auto result = dashboard_service_->get_metric_time_series("cpu_utilization", 24);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value().metric_name, "cpu_utilization");
    EXPECT_EQ(result.value().timestamps.size(), 24);
    EXPECT_EQ(result.value().values.size(), 24);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}