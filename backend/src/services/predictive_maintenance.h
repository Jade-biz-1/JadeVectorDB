#ifndef JADEVECTORDB_PREDICTIVE_MAINTENANCE_H
#define JADEVECTORDB_PREDICTIVE_MAINTENANCE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>

#include "services/monitoring_service.h"
#include "services/metrics_service.h"
#include "services/analytics_dashboard.h"
#include "lib/logging.h"

namespace jadevectordb {
namespace predictive {

    // Enum for prediction confidence levels
    enum class ConfidenceLevel {
        LOW,      // 0-33% confidence
        MEDIUM,   // 34-66% confidence
        HIGH      // 67-100% confidence
    };

    // Enum for maintenance priority levels
    enum class MaintenancePriority {
        LOW,      // Can be scheduled at convenience
        MEDIUM,   // Should be addressed soon
        HIGH,     // Requires attention within 24 hours
        CRITICAL  // Requires immediate attention
    };

    // Structure for resource utilization predictions
    struct ResourcePrediction {
        std::string resource_type;  // "cpu", "memory", "disk", "network"
        std::string resource_id;    // Specific resource identifier (e.g., server name)
        double current_utilization;
        double predicted_utilization;
        std::chrono::system_clock::time_point prediction_time;
        std::chrono::system_clock::time_point predicted_exhaustion_time;
        ConfidenceLevel confidence;
        double confidence_percentage;
        MaintenancePriority priority;
        std::string recommendation;  // Recommended action
        
        ResourcePrediction() : current_utilization(0.0), predicted_utilization(0.0), 
                              confidence(ConfidenceLevel::LOW), confidence_percentage(0.0),
                              priority(MaintenancePriority::LOW) {}
    };

    // Structure for performance degradation predictions
    struct PerformanceDegradationPrediction {
        std::string component;     // Component that may degrade
        std::string metric_name;    // Metric being monitored
        double current_value;
        double predicted_value;
        std::chrono::system_clock::time_point prediction_time;
        std::chrono::system_clock::time_point predicted_degradation_time;
        ConfidenceLevel confidence;
        double confidence_percentage;
        MaintenancePriority priority;
        std::string recommendation;
        std::vector<std::string> affected_services;  // Services that may be affected
        
        PerformanceDegradationPrediction() : current_value(0.0), predicted_value(0.0),
                                           confidence(ConfidenceLevel::LOW), confidence_percentage(0.0),
                                           priority(MaintenancePriority::LOW) {}
    };

    // Structure for scaling recommendations
    struct ScalingRecommendation {
        std::string resource_type;  // "cpu", "memory", "storage"
        std::string action;          // "scale_up", "scale_down", "maintain"
        int current_instances;
        int recommended_instances;
        double utilization_threshold;
        std::chrono::system_clock::time_point recommendation_time;
        ConfidenceLevel confidence;
        double confidence_percentage;
        MaintenancePriority priority;
        std::string justification;  // Reason for recommendation
        std::unordered_map<std::string, std::string> parameters;  // Additional parameters for scaling
        
        ScalingRecommendation() : current_instances(0), recommended_instances(0),
                                 utilization_threshold(0.0), confidence(ConfidenceLevel::LOW),
                                 confidence_percentage(0.0), priority(MaintenancePriority::LOW) {}
    };

    // Structure for capacity planning projections
    struct CapacityProjection {
        std::string resource_type;   // "storage", "compute", "memory"
        std::string timeframe;       // "30_days", "90_days", "1_year"
        double current_capacity;
        double projected_capacity_needed;
        double growth_rate_percentage;  // Annual growth rate
        std::chrono::system_clock::time_point projection_time;
        ConfidenceLevel confidence;
        double confidence_percentage;
        std::string recommendation;
        std::chrono::system_clock::time_point estimated_exhaustion_time;
        
        CapacityProjection() : current_capacity(0.0), projected_capacity_needed(0.0),
                               growth_rate_percentage(0.0), confidence(ConfidenceLevel::LOW),
                               confidence_percentage(0.0) {}
    };

    // Structure for maintenance alerts
    struct MaintenanceAlert {
        std::string id;
        std::string title;
        std::string description;
        MaintenancePriority priority;
        std::chrono::system_clock::time_point created_at;
        std::chrono::system_clock::time_point predicted_occurrence;
        std::vector<std::string> affected_components;
        std::string recommended_action;
        bool acknowledged;
        std::string acknowledged_by;
        std::chrono::system_clock::time_point acknowledged_at;
        std::unordered_map<std::string, std::string> additional_data;
        
        MaintenanceAlert() : priority(MaintenancePriority::LOW), acknowledged(false) {}
    };

    // Configuration for predictive maintenance
    struct PredictiveMaintenanceConfig {
        bool enabled = true;
        int prediction_interval_seconds = 300;  // How often to make predictions (5 minutes)
        int data_collection_window_hours = 24;   // How much historical data to use
        double cpu_utilization_threshold = 80.0; // Threshold for CPU alerts (%)
        double memory_utilization_threshold = 85.0; // Threshold for memory alerts (%)
        double disk_utilization_threshold = 90.0; // Threshold for disk alerts (%)
        double network_utilization_threshold = 75.0; // Threshold for network alerts (%)
        int forecast_horizon_hours = 72;         // How far ahead to predict (3 days)
        std::string notification_channels = "log,email,slack"; // Where to send alerts
        bool enable_automatic_recommendations = true; // Whether to generate automatic recommendations
        int recommendation_review_period_hours = 168; // How often to review recommendations (1 week)
        
        PredictiveMaintenanceConfig() = default;
    };

    // Interface for prediction algorithms
    class IPredictionAlgorithm {
    public:
        virtual ~IPredictionAlgorithm() = default;
        
        /**
         * @brief Make a prediction based on historical data
         * @param historical_data Time series of historical values
         * @param prediction_horizon Hours into the future to predict
         * @return Predicted value and confidence level
         */
        virtual std::pair<double, ConfidenceLevel> predict(
            const std::vector<std::pair<std::chrono::system_clock::time_point, double>>& historical_data,
            int prediction_horizon_hours) = 0;
        
        /**
         * @brief Get the name of the algorithm
         */
        virtual std::string get_algorithm_name() const = 0;
        
        /**
         * @brief Get algorithm-specific parameters
         */
        virtual std::unordered_map<std::string, std::string> get_parameters() const = 0;
        
        /**
         * @brief Set algorithm-specific parameters
         */
        virtual void set_parameters(const std::unordered_map<std::string, std::string>& params) = 0;
    };

    // Linear regression prediction algorithm
    class LinearRegressionPrediction : public IPredictionAlgorithm {
    private:
        std::unordered_map<std::string, std::string> parameters_;
        
    public:
        LinearRegressionPrediction();
        
        std::pair<double, ConfidenceLevel> predict(
            const std::vector<std::pair<std::chrono::system_clock::time_point, double>>& historical_data,
            int prediction_horizon_hours) override;
        
        std::string get_algorithm_name() const override;
        
        std::unordered_map<std::string, std::string> get_parameters() const override;
        
        void set_parameters(const std::unordered_map<std::string, std::string>& params) override;
        
    private:
        double calculate_slope_and_intercept(
            const std::vector<std::pair<std::chrono::system_clock::time_point, double>>& data,
            double& slope, double& intercept) const;
    };

    // Exponential smoothing prediction algorithm
    class ExponentialSmoothingPrediction : public IPredictionAlgorithm {
    private:
        std::unordered_map<std::string, std::string> parameters_;
        double smoothing_factor_;  // Alpha parameter (0.0 to 1.0)
        
    public:
        ExponentialSmoothingPrediction(double smoothing_factor = 0.3);
        
        std::pair<double, ConfidenceLevel> predict(
            const std::vector<std::pair<std::chrono::system_clock::time_point, double>>& historical_data,
            int prediction_horizon_hours) override;
        
        std::string get_algorithm_name() const override;
        
        std::unordered_map<std::string, std::string> get_parameters() const override;
        
        void set_parameters(const std::unordered_map<std::string, std::string>& params) override;
    };

    // Resource exhaustion prediction service
    class ResourceExhaustionPredictor {
    private:
        std::shared_ptr<MetricsService> metrics_service_;
        std::shared_ptr<MonitoringService> monitoring_service_;
        std::shared_ptr<logging::Logger> logger_;
        std::unique_ptr<IPredictionAlgorithm> prediction_algorithm_;
        PredictiveMaintenanceConfig config_;
        
    public:
        ResourceExhaustionPredictor(
            std::shared_ptr<MetricsService> metrics_service,
            std::shared_ptr<MonitoringService> monitoring_service,
            std::unique_ptr<IPredictionAlgorithm> prediction_algorithm = nullptr);
        
        ~ResourceExhaustionPredictor() = default;
        
        /**
         * @brief Predict resource exhaustion for all monitored resources
         * @return Vector of resource predictions
         */
        Result<std::vector<ResourcePrediction>> predict_resource_exhaustion();
        
        /**
         * @brief Predict exhaustion for a specific resource
         * @param resource_type Type of resource to predict
         * @param resource_id Specific resource identifier
         * @return Resource prediction
         */
        Result<ResourcePrediction> predict_resource_exhaustion(
            const std::string& resource_type,
            const std::string& resource_id = "");
        
        /**
         * @brief Configure the predictor with specific settings
         * @param config Configuration settings
         * @return True if configuration was successful
         */
        Result<void> configure(const PredictiveMaintenanceConfig& config);
        
        /**
         * @brief Get current configuration
         */
        const PredictiveMaintenanceConfig& get_config() const;
        
        /**
         * @brief Change the prediction algorithm
         * @param algorithm New prediction algorithm
         */
        void set_prediction_algorithm(std::unique_ptr<IPredictionAlgorithm> algorithm);
        
    private:
        /**
         * @brief Get historical data for a specific metric
         * @param metric_name Name of the metric
         * @param hours Hours of historical data to retrieve
         * @return Time series data
         */
        Result<std::vector<std::pair<std::chrono::system_clock::time_point, double>>> 
        get_historical_data(const std::string& metric_name, int hours);
        
        /**
         * @brief Calculate when a resource will be exhausted based on current trend
         * @param current_utilization Current utilization percentage
         * @param trend_rate Rate of change per hour
         * @return Predicted exhaustion time
         */
        std::chrono::system_clock::time_point calculate_exhaustion_time(
            double current_utilization, double trend_rate) const;
        
        /**
         * @brief Generate recommendation based on prediction
         * @param prediction Resource prediction
         * @return Recommendation string
         */
        std::string generate_recommendation(const ResourcePrediction& prediction) const;
    };

    // Performance degradation forecaster
    class PerformanceDegradationForecaster {
    private:
        std::shared_ptr<MetricsService> metrics_service_;
        std::shared_ptr<MonitoringService> monitoring_service_;
        std::shared_ptr<logging::Logger> logger_;
        std::unique_ptr<IPredictionAlgorithm> prediction_algorithm_;
        PredictiveMaintenanceConfig config_;
        
    public:
        PerformanceDegradationForecaster(
            std::shared_ptr<MetricsService> metrics_service,
            std::shared_ptr<MonitoringService> monitoring_service,
            std::unique_ptr<IPredictionAlgorithm> prediction_algorithm = nullptr);
        
        ~PerformanceDegradationForecaster() = default;
        
        /**
         * @brief Forecast performance degradation for all components
         * @return Vector of performance degradation predictions
         */
        Result<std::vector<PerformanceDegradationPrediction>> forecast_performance_degradation();
        
        /**
         * @brief Forecast degradation for a specific component
         * @param component Component to forecast
         * @param metric_name Specific metric to monitor
         * @return Performance degradation prediction
         */
        Result<PerformanceDegradationPrediction> forecast_degradation(
            const std::string& component,
            const std::string& metric_name);
        
        /**
         * @brief Configure the forecaster with specific settings
         * @param config Configuration settings
         * @return True if configuration was successful
         */
        Result<void> configure(const PredictiveMaintenanceConfig& config);
        
        /**
         * @brief Get current configuration
         */
        const PredictiveMaintenanceConfig& get_config() const;
        
        /**
         * @brief Change the prediction algorithm
         * @param algorithm New prediction algorithm
         */
        void set_prediction_algorithm(std::unique_ptr<IPredictionAlgorithm> algorithm);
        
    private:
        /**
         * @brief Get historical performance data
         * @param component Component to analyze
         * @param metric_name Metric to analyze
         * @param hours Hours of historical data to retrieve
         * @return Time series data
         */
        Result<std::vector<std::pair<std::chrono::system_clock::time_point, double>>> 
        get_performance_data(const std::string& component, const std::string& metric_name, int hours);
        
        /**
         * @brief Identify services affected by component degradation
         * @param component Component that may degrade
         * @return List of affected services
         */
        std::vector<std::string> identify_affected_services(const std::string& component) const;
        
        /**
         * @brief Generate recommendation based on forecast
         * @param forecast Performance degradation forecast
         * @return Recommendation string
         */
        std::string generate_recommendation(const PerformanceDegradationPrediction& forecast) const;
    };

} // namespace predictive
} // namespace jadevectordb

#endif // JADEVECTORDB_PREDICTIVE_MAINTENANCE_H