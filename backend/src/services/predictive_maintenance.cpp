#include "predictive_maintenance.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace jadevectordb {
namespace predictive {

// LinearRegressionPrediction implementation
LinearRegressionPrediction::LinearRegressionPrediction() {
    parameters_["algorithm"] = "linear_regression";
    parameters_["description"] = "Simple linear regression for trend prediction";
}

std::pair<double, ConfidenceLevel> LinearRegressionPrediction::predict(
    const std::vector<std::pair<std::chrono::system_clock::time_point, double>>& historical_data,
    int prediction_horizon_hours) {
    
    if (historical_data.size() < 2) {
        return {0.0, ConfidenceLevel::LOW};
    }
    
    double slope, intercept;
    double r_squared = calculate_slope_and_intercept(historical_data, slope, intercept);
    
    // Calculate prediction time point
    auto last_time = historical_data.back().first;
    auto prediction_time = last_time + std::chrono::hours(prediction_horizon_hours);
    
    // Convert time to hours since epoch for calculation
    auto epoch = std::chrono::system_clock::time_point();
    auto prediction_duration = std::chrono::duration_cast<std::chrono::hours>(
        prediction_time - epoch).count();
    
    double predicted_value = slope * prediction_duration + intercept;
    
    // Clamp predicted value to reasonable bounds (0-100 for percentages)
    predicted_value = std::max(0.0, std::min(100.0, predicted_value));
    
    // Determine confidence based on R-squared value
    ConfidenceLevel confidence;
    if (r_squared >= 0.8) {
        confidence = ConfidenceLevel::HIGH;
    } else if (r_squared >= 0.5) {
        confidence = ConfidenceLevel::MEDIUM;
    } else {
        confidence = ConfidenceLevel::LOW;
    }
    
    return {predicted_value, confidence};
}

std::string LinearRegressionPrediction::get_algorithm_name() const {
    return "Linear Regression";
}

std::unordered_map<std::string, std::string> LinearRegressionPrediction::get_parameters() const {
    return parameters_;
}

void LinearRegressionPrediction::set_parameters(const std::unordered_map<std::string, std::string>& params) {
    parameters_ = params;
}

double LinearRegressionPrediction::calculate_slope_and_intercept(
    const std::vector<std::pair<std::chrono::system_clock::time_point, double>>& data,
    double& slope, double& intercept) const {
    
    // Convert time points to hours since epoch
    std::vector<double> x_values(data.size());
    std::vector<double> y_values(data.size());
    
    auto epoch = std::chrono::system_clock::time_point();
    for (size_t i = 0; i < data.size(); ++i) {
        x_values[i] = std::chrono::duration_cast<std::chrono::hours>(
            data[i].first - epoch).count();
        y_values[i] = data[i].second;
    }
    
    // Calculate means
    double x_mean = std::accumulate(x_values.begin(), x_values.end(), 0.0) / x_values.size();
    double y_mean = std::accumulate(y_values.begin(), y_values.end(), 0.0) / y_values.size();
    
    // Calculate slope and intercept using least squares method
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (size_t i = 0; i < x_values.size(); ++i) {
        double x_diff = x_values[i] - x_mean;
        double y_diff = y_values[i] - y_mean;
        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }
    
    if (denominator == 0.0) {
        slope = 0.0;
        intercept = y_mean;
        return 0.0; // R-squared = 0
    }
    
    slope = numerator / denominator;
    intercept = y_mean - slope * x_mean;
    
    // Calculate R-squared
    double ss_res = 0.0; // Residual sum of squares
    double ss_tot = 0.0; // Total sum of squares
    
    for (size_t i = 0; i < x_values.size(); ++i) {
        double y_pred = slope * x_values[i] + intercept;
        ss_res += (y_values[i] - y_pred) * (y_values[i] - y_pred);
        ss_tot += (y_values[i] - y_mean) * (y_values[i] - y_mean);
    }
    
    double r_squared = (ss_tot != 0.0) ? 1.0 - (ss_res / ss_tot) : 0.0;
    
    return r_squared;
}

// ExponentialSmoothingPrediction implementation
ExponentialSmoothingPrediction::ExponentialSmoothingPrediction(double smoothing_factor)
    : smoothing_factor_(smoothing_factor) {
    parameters_["algorithm"] = "exponential_smoothing";
    parameters_["smoothing_factor"] = std::to_string(smoothing_factor);
    parameters_["description"] = "Exponential smoothing for trend prediction";
}

std::pair<double, ConfidenceLevel> ExponentialSmoothingPrediction::predict(
    const std::vector<std::pair<std::chrono::system_clock::time_point, double>>& historical_data,
    int prediction_horizon_hours) {
    
    if (historical_data.empty()) {
        return {0.0, ConfidenceLevel::LOW};
    }
    
    if (historical_data.size() == 1) {
        return {historical_data.front().second, ConfidenceLevel::LOW};
    }
    
    // Apply exponential smoothing
    double smoothed_value = historical_data.front().second;
    
    for (size_t i = 1; i < historical_data.size(); ++i) {
        smoothed_value = smoothing_factor_ * historical_data[i].second + 
                         (1.0 - smoothing_factor_) * smoothed_value;
    }
    
    // For exponential smoothing, we assume the trend continues
    // This is a simplification - in practice, we might calculate the trend
    double predicted_value = smoothed_value;
    
    // If we have enough data points, we can calculate a simple trend
    if (historical_data.size() >= 3) {
        double first_third_avg = 0.0;
        double last_third_avg = 0.0;
        size_t third_size = historical_data.size() / 3;
        
        // Calculate average of first third
        for (size_t i = 0; i < third_size; ++i) {
            first_third_avg += historical_data[i].second;
        }
        first_third_avg /= third_size;
        
        // Calculate average of last third
        for (size_t i = historical_data.size() - third_size; i < historical_data.size(); ++i) {
            last_third_avg += historical_data[i].second;
        }
        last_third_avg /= third_size;
        
        // Calculate trend
        double trend = (last_third_avg - first_third_avg) / (historical_data.size() / 3);
        
        // Apply trend to prediction
        predicted_value = smoothed_value + trend * prediction_horizon_hours;
    }
    
    // Clamp predicted value to reasonable bounds (0-100 for percentages)
    predicted_value = std::max(0.0, std::min(100.0, predicted_value));
    
    // Determine confidence based on data stability
    double variance = 0.0;
    double mean = 0.0;
    
    // Calculate mean
    for (const auto& data_point : historical_data) {
        mean += data_point.second;
    }
    mean /= historical_data.size();
    
    // Calculate variance
    for (const auto& data_point : historical_data) {
        variance += (data_point.second - mean) * (data_point.second - mean);
    }
    variance /= historical_data.size();
    
    double std_dev = std::sqrt(variance);
    
    // Determine confidence based on standard deviation
    ConfidenceLevel confidence;
    if (std_dev < 5.0) {  // Very stable data
        confidence = ConfidenceLevel::HIGH;
    } else if (std_dev < 15.0) {  // Moderately stable data
        confidence = ConfidenceLevel::MEDIUM;
    } else {  // Highly variable data
        confidence = ConfidenceLevel::LOW;
    }
    
    return {predicted_value, confidence};
}

std::string ExponentialSmoothingPrediction::get_algorithm_name() const {
    return "Exponential Smoothing";
}

std::unordered_map<std::string, std::string> ExponentialSmoothingPrediction::get_parameters() const {
    return parameters_;
}

void ExponentialSmoothingPrediction::set_parameters(const std::unordered_map<std::string, std::string>& params) {
    parameters_ = params;
    auto it = parameters_.find("smoothing_factor");
    if (it != parameters_.end()) {
        try {
            smoothing_factor_ = std::stod(it->second);
        } catch (...) {
            // Keep default value if parsing fails
        }
    }
}

// ResourceExhaustionPredictor implementation
ResourceExhaustionPredictor::ResourceExhaustionPredictor(
    std::shared_ptr<MetricsService> metrics_service,
    std::shared_ptr<MonitoringService> monitoring_service,
    std::unique_ptr<IPredictionAlgorithm> prediction_algorithm)
    : metrics_service_(metrics_service)
    , monitoring_service_(monitoring_service)
    , prediction_algorithm_(std::move(prediction_algorithm)) {
    
    logger_ = logging::LoggerManager::get_logger("ResourceExhaustionPredictor");
    
    // Default to linear regression if no algorithm provided
    if (!prediction_algorithm_) {
        prediction_algorithm_ = std::make_unique<LinearRegressionPrediction>();
    }
}

Result<std::vector<ResourcePrediction>> ResourceExhaustionPredictor::predict_resource_exhaustion() {
    std::vector<ResourcePrediction> predictions;
    
    // Predict for CPU utilization
    auto cpu_prediction = predict_resource_exhaustion("cpu");
    if (cpu_prediction.has_value()) {
        predictions.push_back(cpu_prediction.value());
    }
    
    // Predict for memory utilization
    auto memory_prediction = predict_resource_exhaustion("memory");
    if (memory_prediction.has_value()) {
        predictions.push_back(memory_prediction.value());
    }
    
    // Predict for disk utilization
    auto disk_prediction = predict_resource_exhaustion("disk");
    if (disk_prediction.has_value()) {
        predictions.push_back(disk_prediction.value());
    }
    
    // Predict for network utilization
    auto network_prediction = predict_resource_exhaustion("network");
    if (network_prediction.has_value()) {
        predictions.push_back(network_prediction.value());
    }
    
    return Result<std::vector<ResourcePrediction>>::success(predictions);
}

Result<ResourcePrediction> ResourceExhaustionPredictor::predict_resource_exhaustion(
    const std::string& resource_type,
    const std::string& resource_id) {
    
    ResourcePrediction prediction;
    prediction.resource_type = resource_type;
    prediction.resource_id = resource_id.empty() ? "system_wide" : resource_id;
    prediction.prediction_time = std::chrono::system_clock::now();
    
    // Get current utilization for the resource
    std::string metric_name;
    double threshold = 0.0;
    
    if (resource_type == "cpu") {
        metric_name = "cpu_utilization";
        threshold = config_.cpu_utilization_threshold;
    } else if (resource_type == "memory") {
        metric_name = "memory_utilization";
        threshold = config_.memory_utilization_threshold;
    } else if (resource_type == "disk") {
        metric_name = "disk_utilization";
        threshold = config_.disk_utilization_threshold;
    } else if (resource_type == "network") {
        metric_name = "network_utilization";
        threshold = config_.network_utilization_threshold;
    } else {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Unknown resource type: " + resource_type);
    }
    
    // Get current metric value
    auto current_value_result = metrics_service_->get_metric_value(metric_name);
    if (current_value_result.has_value()) {
        prediction.current_utilization = current_value_result.value();
    } else {
        prediction.current_utilization = 0.0;
    }
    
    // Get historical data for trend analysis
    auto historical_data = get_historical_data(metric_name, config_.data_collection_window_hours);
    if (!historical_data.has_value()) {
        prediction.predicted_utilization = prediction.current_utilization;
        prediction.confidence = ConfidenceLevel::LOW;
        prediction.confidence_percentage = 0.0;
        prediction.recommendation = "Insufficient data for prediction";
        return Result<ResourcePrediction>::success(prediction);
    }
    
    // Make prediction using the algorithm
    auto prediction_result = prediction_algorithm_->predict(
        historical_data.value(), config_.forecast_horizon_hours);
    
    prediction.predicted_utilization = prediction_result.first;
    prediction.confidence = prediction_result.second;
    
    // Calculate confidence percentage
    switch (prediction.confidence) {
        case ConfidenceLevel::HIGH:
            prediction.confidence_percentage = 90.0;
            break;
        case ConfidenceLevel::MEDIUM:
            prediction.confidence_percentage = 60.0;
            break;
        case ConfidenceLevel::LOW:
        default:
            prediction.confidence_percentage = 30.0;
            break;
    }
    
    // Calculate predicted exhaustion time if approaching threshold
    if (prediction.predicted_utilization > threshold) {
        // Calculate trend rate from historical data
        if (historical_data.value().size() >= 2) {
            auto first_point = historical_data.value().front();
            auto last_point = historical_data.value().back();
            auto time_diff = std::chrono::duration_cast<std::chrono::hours>(
                last_point.first - first_point.first).count();
            
            if (time_diff > 0) {
                double utilization_diff = last_point.second - first_point.second;
                double trend_rate = utilization_diff / time_diff; // % per hour
                
                if (trend_rate > 0) {
                    // Calculate time until threshold is reached
                    double utilization_to_threshold = threshold - prediction.current_utilization;
                    double hours_until_threshold = utilization_to_threshold / trend_rate;
                    
                    if (hours_until_threshold > 0) {
                        prediction.predicted_exhaustion_time = 
                            std::chrono::system_clock::now() + std::chrono::hours(
                                static_cast<int>(std::round(hours_until_threshold)));
                    }
                }
            }
        }
    }
    
    // Set priority based on predicted utilization and confidence
    if (prediction.predicted_utilization > 95.0) {
        prediction.priority = MaintenancePriority::CRITICAL;
    } else if (prediction.predicted_utilization > 90.0) {
        prediction.priority = MaintenancePriority::HIGH;
    } else if (prediction.predicted_utilization > 80.0) {
        prediction.priority = MaintenancePriority::MEDIUM;
    } else {
        prediction.priority = MaintenancePriority::LOW;
    }
    
    // Generate recommendation
    prediction.recommendation = generate_recommendation(prediction);
    
    return Result<ResourcePrediction>::success(prediction);
}

Result<void> ResourceExhaustionPredictor::configure(const PredictiveMaintenanceConfig& config) {
    config_ = config;
    return Result<void>::success();
}

const PredictiveMaintenanceConfig& ResourceExhaustionPredictor::get_config() const {
    return config_;
}

void ResourceExhaustionPredictor::set_prediction_algorithm(std::unique_ptr<IPredictionAlgorithm> algorithm) {
    prediction_algorithm_ = std::move(algorithm);
}

Result<std::vector<std::pair<std::chrono::system_clock::time_point, double>>> 
ResourceExhaustionPredictor::get_historical_data(const std::string& metric_name, int hours) {
    // In a real implementation, we would retrieve historical data from the metrics service
    // For this implementation, we'll generate synthetic data for demonstration
    
    std::vector<std::pair<std::chrono::system_clock::time_point, double>> historical_data;
    
    auto now = std::chrono::system_clock::now();
    auto start_time = now - std::chrono::hours(hours);
    
    // Generate synthetic data with some trend
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> noise(-5.0, 5.0);
    
    for (int i = 0; i < hours; ++i) {
        auto timestamp = start_time + std::chrono::hours(i);
        
        // Create a synthetic trend (increasing over time)
        double base_value = 30.0 + (i * 0.5); // 0.5% increase per hour
        double noisy_value = base_value + noise(gen);
        
        // Clamp to realistic values (0-100%)
        noisy_value = std::max(0.0, std::min(100.0, noisy_value));
        
        historical_data.emplace_back(timestamp, noisy_value);
    }
    
    return Result<std::vector<std::pair<std::chrono::system_clock::time_point, double>>>::success(historical_data);
}

std::chrono::system_clock::time_point ResourceExhaustionPredictor::calculate_exhaustion_time(
    double current_utilization, double trend_rate) const {
    
    if (trend_rate <= 0) {
        // If trend is not increasing, no exhaustion predicted
        return std::chrono::system_clock::time_point::max();
    }
    
    // Calculate time until 100% utilization
    double utilization_to_full = 100.0 - current_utilization;
    double hours_until_full = utilization_to_full / trend_rate;
    
    auto now = std::chrono::system_clock::now();
    return now + std::chrono::hours(static_cast<int>(std::round(hours_until_full)));
}

std::string ResourceExhaustionPredictor::generate_recommendation(const ResourcePrediction& prediction) const {
    std::string recommendation;
    
    if (prediction.predicted_utilization > 95.0) {
        recommendation = "CRITICAL: Resource exhaustion imminent. Immediate action required. "
                        "Consider emergency scaling or workload redistribution.";
    } else if (prediction.predicted_utilization > 90.0) {
        recommendation = "HIGH: Resource utilization approaching critical levels. "
                         "Plan for scaling operations within 24-48 hours.";
    } else if (prediction.predicted_utilization > 80.0) {
        recommendation = "MEDIUM: Resource utilization trending upward. "
                         "Monitor closely and prepare scaling plans.";
    } else if (prediction.predicted_utilization > 70.0) {
        recommendation = "LOW: Resource utilization within normal range. "
                         "Continue routine monitoring.";
    } else {
        recommendation = "NORMAL: Resource utilization is healthy. "
                         "No immediate action required.";
    }
    
    // Add specific recommendations based on resource type
    if (prediction.resource_type == "cpu") {
        recommendation += " For CPU, consider adding more worker nodes or optimizing queries.";
    } else if (prediction.resource_type == "memory") {
        recommendation += " For memory, consider increasing RAM allocation or optimizing memory usage patterns.";
    } else if (prediction.resource_type == "disk") {
        recommendation += " For disk, consider adding storage capacity or implementing data archiving.";
    } else if (prediction.resource_type == "network") {
        recommendation += " For network, consider bandwidth upgrades or traffic optimization.";
    }
    
    return recommendation;
}

// PerformanceDegradationForecaster implementation
PerformanceDegradationForecaster::PerformanceDegradationForecaster(
    std::shared_ptr<MetricsService> metrics_service,
    std::shared_ptr<MonitoringService> monitoring_service,
    std::unique_ptr<IPredictionAlgorithm> prediction_algorithm)
    : metrics_service_(metrics_service)
    , monitoring_service_(monitoring_service)
    , prediction_algorithm_(std::move(prediction_algorithm)) {
    
    logger_ = logging::LoggerManager::get_logger("PerformanceDegradationForecaster");
    
    // Default to linear regression if no algorithm provided
    if (!prediction_algorithm_) {
        prediction_algorithm_ = std::make_unique<LinearRegressionPrediction>();
    }
}

Result<std::vector<PerformanceDegradationPrediction>> PerformanceDegradationForecaster::forecast_performance_degradation() {
    std::vector<PerformanceDegradationPrediction> forecasts;
    
    // Forecast for query response time
    auto query_forecast = forecast_degradation("query_processor", "query_response_time_ms");
    if (query_forecast.has_value()) {
        forecasts.push_back(query_forecast.value());
    }
    
    // Forecast for similarity search latency
    auto search_forecast = forecast_degradation("similarity_search", "search_latency_ms");
    if (search_forecast.has_value()) {
        forecasts.push_back(search_forecast.value());
    }
    
    // Forecast for vector insertion rate
    auto insert_forecast = forecast_degradation("vector_storage", "insert_rate_per_second");
    if (insert_forecast.has_value()) {
        forecasts.push_back(insert_forecast.value());
    }
    
    // Forecast for index build time
    auto index_forecast = forecast_degradation("index_builder", "index_build_time_ms");
    if (index_forecast.has_value()) {
        forecasts.push_back(index_forecast.value());
    }
    
    return Result<std::vector<PerformanceDegradationPrediction>>::success(forecasts);
}

Result<PerformanceDegradationPrediction> PerformanceDegradationForecaster::forecast_degradation(
    const std::string& component,
    const std::string& metric_name) {
    
    PerformanceDegradationPrediction forecast;
    forecast.component = component;
    forecast.metric_name = metric_name;
    forecast.prediction_time = std::chrono::system_clock::now();
    
    // Get current metric value
    auto current_value_result = metrics_service_->get_metric_value(metric_name);
    if (current_value_result.has_value()) {
        forecast.current_value = current_value_result.value();
    } else {
        forecast.current_value = 0.0;
    }
    
    // Get historical data for trend analysis
    auto historical_data = get_performance_data(component, metric_name, config_.data_collection_window_hours);
    if (!historical_data.has_value()) {
        forecast.predicted_value = forecast.current_value;
        forecast.confidence = ConfidenceLevel::LOW;
        forecast.confidence_percentage = 0.0;
        forecast.recommendation = "Insufficient data for forecasting";
        return Result<PerformanceDegradationPrediction>::success(forecast);
    }
    
    // Make prediction using the algorithm
    auto prediction_result = prediction_algorithm_->predict(
        historical_data.value(), config_.forecast_horizon_hours);
    
    forecast.predicted_value = prediction_result.first;
    forecast.confidence = prediction_result.second;
    
    // Calculate confidence percentage
    switch (forecast.confidence) {
        case ConfidenceLevel::HIGH:
            forecast.confidence_percentage = 90.0;
            break;
        case ConfidenceLevel::MEDIUM:
            forecast.confidence_percentage = 60.0;
            break;
        case ConfidenceLevel::LOW:
        default:
            forecast.confidence_percentage = 30.0;
            break;
    }
    
    // Calculate predicted degradation time if performance is worsening
    if (forecast.predicted_value > forecast.current_value) {
        // Calculate trend rate from historical data
        if (historical_data.value().size() >= 2) {
            auto first_point = historical_data.value().front();
            auto last_point = historical_data.value().back();
            auto time_diff = std::chrono::duration_cast<std::chrono::hours>(
                last_point.first - first_point.first).count();
            
            if (time_diff > 0) {
                double value_diff = last_point.second - first_point.second;
                double trend_rate = value_diff / time_diff; // units per hour
                
                if (trend_rate > 0) {
                    // Calculate time until significant degradation (2x current value)
                    double value_to_degradation = forecast.current_value; // Degradation when doubles
                    double hours_until_degradation = value_to_degradation / trend_rate;
                    
                    if (hours_until_degradation > 0) {
                        forecast.predicted_degradation_time = 
                            std::chrono::system_clock::now() + std::chrono::hours(
                                static_cast<int>(std::round(hours_until_degradation)));
                    }
                }
            }
        }
    }
    
    // Set priority based on predicted degradation and confidence
    double degradation_ratio = forecast.predicted_value / std::max(1.0, forecast.current_value);
    
    if (degradation_ratio > 2.0) {
        forecast.priority = MaintenancePriority::CRITICAL;
    } else if (degradation_ratio > 1.5) {
        forecast.priority = MaintenancePriority::HIGH;
    } else if (degradation_ratio > 1.2) {
        forecast.priority = MaintenancePriority::MEDIUM;
    } else {
        forecast.priority = MaintenancePriority::LOW;
    }
    
    // Identify affected services
    forecast.affected_services = identify_affected_services(component);
    
    // Generate recommendation
    forecast.recommendation = generate_recommendation(forecast);
    
    return Result<PerformanceDegradationPrediction>::success(forecast);
}

Result<void> PerformanceDegradationForecaster::configure(const PredictiveMaintenanceConfig& config) {
    config_ = config;
    return Result<void>::success();
}

const PredictiveMaintenanceConfig& PerformanceDegradationForecaster::get_config() const {
    return config_;
}

void PerformanceDegradationForecaster::set_prediction_algorithm(std::unique_ptr<IPredictionAlgorithm> algorithm) {
    prediction_algorithm_ = std::move(algorithm);
}

Result<std::vector<std::pair<std::chrono::system_clock::time_point, double>>> 
PerformanceDegradationForecaster::get_performance_data(const std::string& component, const std::string& metric_name, int hours) {
    // In a real implementation, we would retrieve historical data from the metrics service
    // For this implementation, we'll generate synthetic data for demonstration
    
    std::vector<std::pair<std::chrono::system_clock::time_point, double>> historical_data;
    
    auto now = std::chrono::system_clock::now();
    auto start_time = now - std::chrono::hours(hours);
    
    // Generate synthetic data with some trend
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> noise(-2.0, 2.0);
    
    for (int i = 0; i < hours; ++i) {
        auto timestamp = start_time + std::chrono::hours(i);
        
        // Create a synthetic trend (increasing over time for degradation)
        double base_value = 20.0 + (i * 0.3); // 0.3ms increase per hour
        double noisy_value = base_value + noise(gen);
        
        // Clamp to realistic positive values
        noisy_value = std::max(1.0, noisy_value);
        
        historical_data.emplace_back(timestamp, noisy_value);
    }
    
    return Result<std::vector<std::pair<std::chrono::system_clock::time_point, double>>>::success(historical_data);
}

std::vector<std::string> PerformanceDegradationForecaster::identify_affected_services(const std::string& component) const {
    std::vector<std::string> affected_services;
    
    // Map components to affected services
    if (component == "query_processor") {
        affected_services = {"search_service", "api_gateway", "client_applications"};
    } else if (component == "similarity_search") {
        affected_services = {"vector_search_service", "recommendation_engine", "semantic_search"};
    } else if (component == "vector_storage") {
        affected_services = {"data_ingestion_pipeline", "realtime_vector_updates", "batch_processing"};
    } else if (component == "index_builder") {
        affected_services = {"index_maintenance", "query_performance_optimizer", "data_analytics"};
    } else {
        affected_services = {"all_services"}; // Default when component is unknown
    }
    
    return affected_services;
}

std::string PerformanceDegradationForecaster::generate_recommendation(const PerformanceDegradationPrediction& forecast) const {
    std::string recommendation;
    
    double degradation_ratio = forecast.predicted_value / std::max(1.0, forecast.current_value);
    
    if (degradation_ratio > 2.0) {
        recommendation = "CRITICAL: Severe performance degradation predicted. "
                        "Immediate investigation and remediation required.";
    } else if (degradation_ratio > 1.5) {
        recommendation = "HIGH: Significant performance degradation predicted. "
                         "Plan for performance optimization within 24-48 hours.";
    } else if (degradation_ratio > 1.2) {
        recommendation = "MEDIUM: Moderate performance degradation predicted. "
                         "Monitor closely and prepare optimization plans.";
    } else {
        recommendation = "LOW: Minor performance changes predicted. "
                         "Continue routine monitoring.";
    }
    
    // Add component-specific recommendations
    if (forecast.component == "query_processor") {
        recommendation += " For query processor, consider query optimization, indexing improvements, or adding query cache.";
    } else if (forecast.component == "similarity_search") {
        recommendation += " For similarity search, consider index optimization, quantization, or algorithm tuning.";
    } else if (forecast.component == "vector_storage") {
        recommendation += " For vector storage, consider storage optimization, memory mapping, or compression.";
    } else if (forecast.component == "index_builder") {
        recommendation += " For index builder, consider incremental updates, parallel processing, or resource allocation.";
    }
    
    return recommendation;
}

} // namespace predictive
} // namespace jadevectordb

} // namespace predictive
} // namespace jadevectordb