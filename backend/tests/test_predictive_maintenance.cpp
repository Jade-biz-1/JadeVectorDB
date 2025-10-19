#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <chrono>

#include "services/predictive_maintenance.h"
#include "services/metrics_service.h"
#include "services/monitoring_service.h"
#include "lib/logging.h"

using namespace jadevectordb;
using namespace jadevectordb::predictive;

// Mock classes for testing
class MockMetricsService : public MetricsService {
public:
    MOCK_METHOD(Result<double>, get_metric_value, (const std::string& metric_name), (const, override));
    MOCK_METHOD(Result<void>, record_metric, (const std::string& name, double value, const std::unordered_map<std::string, std::string>& labels), (override));
    MOCK_METHOD(Result<std::vector<Metric>>, get_all_metrics, (), (const, override));
    MOCK_METHOD(Result<std::vector<Metric>>, get_metrics_by_type, (MetricType type), (const, override));
    MOCK_METHOD(Result<std::vector<Metric>>, get_metrics_by_label, (const std::string& label_key, const std::string& label_value), (const, override));
    MOCK_METHOD(Result<std::vector<Metric>>, get_metrics_by_name_pattern, (const std::string& pattern), (const, override));
    MOCK_METHOD(Result<bool>, initialize, (const MetricsConfig& config), (override));
};

class MockMonitoringService : public MonitoringService {
public:
    MOCK_METHOD(Result<SystemHealth>, check_system_health, (), (const, override));
    MOCK_METHOD(Result<DatabaseStatus>, get_database_status, (const std::string& database_id), (const, override));
    MOCK_METHOD(Result<std::vector<DatabaseStatus>>, get_all_database_statuses, (), (const, override));
    MOCK_METHOD(Result<bool>, initialize, (const MonitoringConfig& config), (override));
};

// Test fixture for predictive maintenance components
class PredictiveMaintenanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create mock services
        mock_metrics_service_ = std::make_shared<MockMetricsService>();
        mock_monitoring_service_ = std::make_shared<MockMonitoringService>();
        
        // Create configuration
        config_.enabled = true;
        config_.prediction_interval_seconds = 1;  // Fast for testing
        config_.data_collection_window_hours = 24;
        config_.cpu_utilization_threshold = 80.0;
        config_.memory_utilization_threshold = 85.0;
        config_.disk_utilization_threshold = 90.0;
        config_.network_utilization_threshold = 75.0;
        config_.forecast_horizon_hours = 72;
        config_.notification_channels = "log";
    }
    
    void TearDown() override {
        // Cleanup
        mock_metrics_service_.reset();
        mock_monitoring_service_.reset();
    }
    
    std::shared_ptr<MockMetricsService> mock_metrics_service_;
    std::shared_ptr<MockMonitoringService> mock_monitoring_service_;
    PredictiveMaintenanceConfig config_;
};

// Test LinearRegressionPrediction
TEST_F(PredictiveMaintenanceTest, LinearRegressionPredictionTest) {
    LinearRegressionPrediction predictor;
    
    // Create synthetic data with a clear trend
    std::vector<std::pair<std::chrono::system_clock::time_point, double>> data;
    auto start_time = std::chrono::system_clock::now() - std::chrono::hours(10);
    
    // Create data with linear trend: y = 2x + 10
    for (int i = 0; i < 10; ++i) {
        auto timestamp = start_time + std::chrono::hours(i);
        double value = 2.0 * i + 10.0 + (i * 0.1); // Add small noise
        data.emplace_back(timestamp, value);
    }
    
    // Make prediction for 5 hours ahead
    auto prediction = predictor.predict(data, 5);
    
    // Should have a prediction and reasonable confidence
    EXPECT_GT(prediction.first, 0.0);
    EXPECT_NE(prediction.second, ConfidenceLevel::LOW); // Should have at least medium confidence
    
    // Test algorithm name
    EXPECT_EQ(predictor.get_algorithm_name(), "Linear Regression");
    
    // Test parameters
    auto params = predictor.get_parameters();
    EXPECT_FALSE(params.empty());
    EXPECT_EQ(params.at("algorithm"), "linear_regression");
}

// Test ExponentialSmoothingPrediction
TEST_F(PredictiveMaintenanceTest, ExponentialSmoothingPredictionTest) {
    ExponentialSmoothingPrediction predictor(0.3); // Alpha = 0.3
    
    // Create synthetic data
    std::vector<std::pair<std::chrono::system_clock::time_point, double>> data;
    auto start_time = std::chrono::system_clock::now() - std::chrono::hours(10);
    
    // Create data with some variation
    for (int i = 0; i < 10; ++i) {
        auto timestamp = start_time + std::chrono::hours(i);
        double value = 50.0 + (i * 2.0) + (i % 3); // Some trend with variation
        data.emplace_back(timestamp, value);
    }
    
    // Make prediction for 3 hours ahead
    auto prediction = predictor.predict(data, 3);
    
    // Should have a prediction
    EXPECT_GT(prediction.first, 0.0);
    
    // Test algorithm name
    EXPECT_EQ(predictor.get_algorithm_name(), "Exponential Smoothing");
    
    // Test parameters
    auto params = predictor.get_parameters();
    EXPECT_FALSE(params.empty());
    EXPECT_EQ(params.at("algorithm"), "exponential_smoothing");
}

// Test ResourceExhaustionPredictor
TEST_F(PredictiveMaintenanceTest, ResourceExhaustionPredictorTest) {
    // Create predictor with mock services
    auto predictor = std::make_unique<ResourceExhaustionPredictor>(
        mock_metrics_service_, mock_monitoring_service_);
    
    // Configure the predictor
    auto config_result = predictor->configure(config_);
    EXPECT_TRUE(config_result.has_value());
    
    // Test prediction for CPU resource
    auto prediction_result = predictor->predict_resource_exhaustion("cpu");
    EXPECT_TRUE(prediction_result.has_value());
    
    const auto& prediction = prediction_result.value();
    EXPECT_EQ(prediction.resource_type, "cpu");
    EXPECT_FALSE(prediction.resource_id.empty());
    EXPECT_GT(prediction.current_utilization, 0.0);
    EXPECT_GT(prediction.predicted_utilization, 0.0);
    
    // Test prediction for all resources
    auto all_predictions_result = predictor->predict_resource_exhaustion();
    EXPECT_TRUE(all_predictions_result.has_value());
    EXPECT_FALSE(all_predictions_result.value().empty());
}

// Test PerformanceDegradationForecaster
TEST_F(PredictiveMaintenanceTest, PerformanceDegradationForecasterTest) {
    // Create forecaster with mock services
    auto forecaster = std::make_unique<PerformanceDegradationForecaster>(
        mock_metrics_service_, mock_monitoring_service_);
    
    // Configure the forecaster
    auto config_result = forecaster->configure(config_);
    EXPECT_TRUE(config_result.has_value());
    
    // Test forecast for query processor
    auto forecast_result = forecaster->forecast_degradation("query_processor", "query_response_time_ms");
    EXPECT_TRUE(forecast_result.has_value());
    
    const auto& forecast = forecast_result.value();
    EXPECT_EQ(forecast.component, "query_processor");
    EXPECT_EQ(forecast.metric_name, "query_response_time_ms");
    EXPECT_GT(forecast.current_value, 0.0);
    EXPECT_GT(forecast.predicted_value, 0.0);
    
    // Test forecast for all components
    auto all_forecasts_result = forecaster->forecast_performance_degradation();
    EXPECT_TRUE(all_forecasts_result.has_value());
    EXPECT_FALSE(all_forecasts_result.value().empty());
}

// Test ScalingRecommendationGenerator
TEST_F(PredictiveMaintenanceTest, ScalingRecommendationGeneratorTest) {
    // Create generator with mock services
    auto generator = std::make_unique<ScalingRecommendationGenerator>(
        mock_metrics_service_, mock_monitoring_service_);
    
    // Configure the generator
    auto config_result = generator->configure(config_);
    EXPECT_TRUE(config_result.has_value());
    
    // Test recommendation for CPU resource
    auto recommendation_result = generator->generate_recommendation("cpu");
    EXPECT_TRUE(recommendation_result.has_value());
    
    const auto& recommendation = recommendation_result.value();
    EXPECT_EQ(recommendation.resource_type, "cpu");
    EXPECT_FALSE(recommendation.action.empty());
    EXPECT_GT(recommendation.current_instances, 0);
    EXPECT_GT(recommendation.recommended_instances, 0);
    
    // Test recommendations for all resources
    auto all_recommendations_result = generator->generate_scaling_recommendations();
    EXPECT_TRUE(all_recommendations_result.has_value());
    EXPECT_FALSE(all_recommendations_result.value().empty());
}

// Test CapacityPlanner
TEST_F(PredictiveMaintenanceTest, CapacityPlannerTest) {
    // Create planner with mock services
    auto planner = std::make_unique<CapacityPlanner>(
        mock_metrics_service_, mock_monitoring_service_);
    
    // Configure the planner
    auto config_result = planner->configure(config_);
    EXPECT_TRUE(config_result.has_value());
    
    // Test projection for storage resource
    auto projection_result = planner->project_capacity("storage");
    EXPECT_TRUE(projection_result.has_value());
    
    const auto& projection = projection_result.value();
    EXPECT_EQ(projection.resource_type, "storage");
    EXPECT_GT(projection.current_capacity, 0.0);
    EXPECT_GT(projection.projected_capacity_needed, 0.0);
    EXPECT_GT(projection.growth_rate_percentage, 0.0);
    
    // Test projections for all resources
    auto all_projections_result = planner->generate_capacity_projections();
    EXPECT_TRUE(all_projections_result.has_value());
    EXPECT_FALSE(all_projections_result.value().empty());
}

// Test PredictiveMaintenanceManager
TEST_F(PredictiveMaintenanceTest, PredictiveMaintenanceManagerTest) {
    // Create manager with mock services
    auto manager = std::make_unique<PredictiveMaintenanceManager>(
        mock_metrics_service_, mock_monitoring_service_);
    
    // Initialize the manager
    auto init_result = manager->initialize(config_);
    EXPECT_TRUE(init_result.has_value());
    
    // Test resource exhaustion predictions
    auto resource_predictions = manager->predict_resource_exhaustion();
    EXPECT_TRUE(resource_predictions.has_value());
    
    // Test performance degradation forecasts
    auto performance_forecasts = manager->forecast_performance_degradation();
    EXPECT_TRUE(performance_forecasts.has_value());
    
    // Test scaling recommendations
    auto scaling_recommendations = manager->generate_scaling_recommendations();
    EXPECT_TRUE(scaling_recommendations.has_value());
    
    // Test capacity projections
    auto capacity_projections = manager->generate_capacity_projections();
    EXPECT_TRUE(capacity_projections.has_value());
    
    // Test maintenance alerts
    auto maintenance_alerts = manager->generate_maintenance_alerts();
    EXPECT_TRUE(maintenance_alerts.has_value());
    
    // Test configuration
    auto config_result = manager->configure(config_);
    EXPECT_TRUE(config_result.has_value());
    
    const auto& retrieved_config = manager->get_config();
    EXPECT_EQ(retrieved_config.enabled, config_.enabled);
    EXPECT_EQ(retrieved_config.prediction_interval_seconds, config_.prediction_interval_seconds);
}

// Test background monitoring
TEST_F(PredictiveMaintenanceTest, BackgroundMonitoringTest) {
    // Create manager with mock services
    auto manager = std::make_unique<PredictiveMaintenanceManager>(
        mock_metrics_service_, mock_monitoring_service_);
    
    // Initialize the manager
    auto init_result = manager->initialize(config_);
    EXPECT_TRUE(init_result.has_value());
    
    // Start background monitoring
    manager->start_background_monitoring();
    
    // Give it a moment to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Stop background monitoring
    manager->stop_background_monitoring();
    
    SUCCEED();
}

// Test algorithm switching
TEST_F(PredictiveMaintenanceTest, AlgorithmSwitchingTest) {
    // Create predictor with linear regression
    auto predictor = std::make_unique<ResourceExhaustionPredictor>(
        mock_metrics_service_, mock_monitoring_service_,
        std::make_unique<LinearRegressionPrediction>());
    
    // Switch to exponential smoothing
    predictor->set_prediction_algorithm(std::make_unique<ExponentialSmoothingPrediction>(0.5));
    
    // Test prediction still works
    auto prediction_result = predictor->predict_resource_exhaustion("cpu");
    EXPECT_TRUE(prediction_result.has_value());
}

// Test confidence levels
TEST_F(PredictiveMaintenanceTest, ConfidenceLevelsTest) {
    LinearRegressionPrediction predictor;
    
    // Test with high-quality data (strong trend)
    std::vector<std::pair<std::chrono::system_clock::time_point, double>> good_data;
    auto start_time = std::chrono::system_clock::now() - std::chrono::hours(10);
    
    // Create data with very clear linear trend
    for (int i = 0; i < 20; ++i) {
        auto timestamp = start_time + std::chrono::hours(i);
        double value = 5.0 * i + 20.0; // Perfect linear trend
        good_data.emplace_back(timestamp, value);
    }
    
    auto good_prediction = predictor.predict(good_data, 5);
    EXPECT_EQ(good_prediction.second, ConfidenceLevel::HIGH);
    
    // Test with low-quality data (no trend)
    std::vector<std::pair<std::chrono::system_clock::time_point, double>> bad_data;
    for (int i = 0; i < 10; ++i) {
        auto timestamp = start_time + std::chrono::hours(i);
        double value = 50.0 + (rand() % 10) - 5; // Random values around 50
        bad_data.emplace_back(timestamp, value);
    }
    
    auto bad_prediction = predictor.predict(bad_data, 5);
    // Even with poor data, should still return a prediction
    EXPECT_GT(bad_prediction.first, 0.0);
}

// Test priority levels
TEST_F(PredictiveMaintenanceTest, PriorityLevelsTest) {
    // Test that high utilization generates high priority alerts
    ResourcePrediction high_utilization_prediction;
    high_utilization_prediction.predicted_utilization = 96.0;
    high_utilization_prediction.priority = MaintenancePriority::CRITICAL;
    
    EXPECT_EQ(high_utilization_prediction.priority, MaintenancePriority::CRITICAL);
    
    // Test that medium utilization generates medium priority alerts
    ResourcePrediction medium_utilization_prediction;
    medium_utilization_prediction.predicted_utilization = 85.0;
    medium_utilization_prediction.priority = MaintenancePriority::HIGH;
    
    EXPECT_EQ(medium_utilization_prediction.priority, MaintenancePriority::HIGH);
    
    // Test that low utilization generates low priority alerts
    ResourcePrediction low_utilization_prediction;
    low_utilization_prediction.predicted_utilization = 45.0;
    low_utilization_prediction.priority = MaintenancePriority::LOW;
    
    EXPECT_EQ(low_utilization_prediction.priority, MaintenancePriority::LOW);
}

// Test maintenance alerts
TEST_F(PredictiveMaintenanceTest, MaintenanceAlertsTest) {
    MaintenanceAlert alert;
    alert.id = "test_alert_123";
    alert.title = "Test Alert";
    alert.description = "This is a test alert";
    alert.priority = MaintenancePriority::MEDIUM;
    alert.created_at = std::chrono::system_clock::now();
    alert.affected_components = {"cpu", "memory"};
    alert.recommended_action = "Increase resources";
    alert.acknowledged = false;
    
    EXPECT_EQ(alert.id, "test_alert_123");
    EXPECT_EQ(alert.title, "Test Alert");
    EXPECT_EQ(alert.priority, MaintenancePriority::MEDIUM);
    EXPECT_FALSE(alert.acknowledged);
    EXPECT_EQ(alert.affected_components.size(), 2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}