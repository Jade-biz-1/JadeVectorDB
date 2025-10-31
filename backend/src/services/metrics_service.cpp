#include "metrics_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>
#include <chrono>

namespace jadevectordb {

MetricsService::MetricsService() {
    logger_ = logging::LoggerManager::get_logger("MetricsService");
}

bool MetricsService::initialize(const MetricsConfig& config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (!validate_config(config)) {
            LOG_ERROR(logger_, "Invalid metrics configuration provided");
            return false;
        }
        
        config_ = config;
        
        LOG_INFO(logger_, "MetricsService initialized with collection_interval: " + 
                std::to_string(config_.collection_interval_seconds) + "s, " +
                "retention_period: " + std::to_string(config_.retention_period_days) + " days");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in MetricsService::initialize: " + std::string(e.what()));
        return false;
    }
}

Result<bool> MetricsService::start_collection() {
    try {
        if (running_) {
            LOG_WARN(logger_, "Metrics collection is already running");
            return true;
        }
        
        LOG_INFO(logger_, "Starting metrics collection");
        
        running_ = true;
        collection_thread_ = std::make_unique<std::thread>(&MetricsService::run_collection_loop, this);
        
        LOG_INFO(logger_, "Metrics collection started successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in start_collection: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to start metrics collection: " + std::string(e.what()));
    }
}

void MetricsService::stop_collection() {
    try {
        if (!running_) {
            LOG_DEBUG(logger_, "Metrics collection is not running");
            return;
        }
        
        LOG_INFO(logger_, "Stopping metrics collection");
        
        running_ = false;
        
        if (collection_thread_ && collection_thread_->joinable()) {
            collection_thread_->join();
        }
        
        collection_thread_.reset();
        
        LOG_INFO(logger_, "Metrics collection stopped successfully");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in stop_collection: " + std::string(e.what()));
    }
}

Result<bool> MetricsService::record_metric(const std::string& metric_name, 
                                       double value, 
                                       const std::unordered_map<std::string, std::string>& labels) {
    try {
        LOG_DEBUG(logger_, "Recording metric " + metric_name + " with value " + std::to_string(value));
        
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        // Create or update the metric
        MetricData metric_data;
        metric_data.name = metric_name;
        metric_data.value = value;
        metric_data.labels = labels;
        metric_data.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Add to metrics map
        metrics_[metric_name].push_back(metric_data);
        
        // Limit metrics history based on retention period
        cleanup_old_metrics(metric_name);
        
        LOG_DEBUG(logger_, "Recorded metric " + metric_name + " successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in record_metric: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to record metric: " + std::string(e.what()));
    }
}

Result<bool> MetricsService::increment_counter(const std::string& counter_name, 
                                          double increment, 
                                          const std::unordered_map<std::string, std::string>& labels) {
    try {
        LOG_DEBUG(logger_, "Incrementing counter " + counter_name + " by " + std::to_string(increment));
        
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        // Get current value or default to 0
        double current_value = 0.0;
        auto it = counters_.find(counter_name);
        if (it != counters_.end()) {
            current_value = it->second;
        }
        
        // Update counter
        counters_[counter_name] = current_value + increment;
        
        // Record as a metric as well
        MetricData metric_data;
        metric_data.name = counter_name;
        metric_data.value = counters_[counter_name];
        metric_data.labels = labels;
        metric_data.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        metric_data.type = "counter";
        
        metrics_[counter_name].push_back(metric_data);
        
        // Limit metrics history based on retention period
        cleanup_old_metrics(counter_name);
        
        LOG_DEBUG(logger_, "Incremented counter " + counter_name + " to " + 
                 std::to_string(counters_[counter_name]));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in increment_counter: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to increment counter: " + std::string(e.what()));
    }
}

Result<bool> MetricsService::update_gauge(const std::string& gauge_name, 
                                     double value, 
                                     const std::unordered_map<std::string, std::string>& labels) {
    try {
        LOG_DEBUG(logger_, "Updating gauge " + gauge_name + " to " + std::to_string(value));
        
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        // Update gauge value
        gauges_[gauge_name] = value;
        
        // Record as a metric as well
        MetricData metric_data;
        metric_data.name = gauge_name;
        metric_data.value = value;
        metric_data.labels = labels;
        metric_data.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        metric_data.type = "gauge";
        
        metrics_[gauge_name].push_back(metric_data);
        
        // Limit metrics history based on retention period
        cleanup_old_metrics(gauge_name);
        
        LOG_DEBUG(logger_, "Updated gauge " + gauge_name + " to " + std::to_string(value));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_gauge: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update gauge: " + std::string(e.what()));
    }
}

Result<bool> MetricsService::record_histogram(const std::string& histogram_name, 
                                         double value, 
                                         const std::unordered_map<std::string, std::string>& labels) {
    try {
        LOG_DEBUG(logger_, "Recording histogram " + histogram_name + " with value " + std::to_string(value));
        
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        // Update histogram buckets
        auto& histogram = histograms_[histogram_name];
        histogram.sum += value;
        histogram.count++;
        
        // Find appropriate bucket and increment
        for (auto& bucket : histogram.buckets) {
            if (value <= bucket.upper_bound) {
                bucket.count++;
                break;
            }
        }
        
        // Record as a metric as well
        MetricData metric_data;
        metric_data.name = histogram_name;
        metric_data.value = value;
        metric_data.labels = labels;
        metric_data.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        metric_data.type = "histogram";
        
        metrics_[histogram_name].push_back(metric_data);
        
        // Limit metrics history based on retention period
        cleanup_old_metrics(histogram_name);
        
        LOG_DEBUG(logger_, "Recorded histogram " + histogram_name + " with value " + std::to_string(value));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in record_histogram: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to record histogram: " + std::string(e.what()));
    }
}

Result<std::vector<MetricData>> MetricsService::get_metrics(const std::string& metric_name) const {
    try {
        std::shared_lock<std::shared_mutex> lock(metrics_mutex_);
        
        auto it = metrics_.find(metric_name);
        if (it != metrics_.end()) {
            return it->second;
        }
        
        LOG_DEBUG(logger_, "No metrics found for name: " + metric_name);
        RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "No metrics found for name: " + metric_name);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get metrics: " + std::string(e.what()));
    }
}

Result<std::unordered_map<std::string, double>> MetricsService::get_all_counters() const {
    try {
        std::shared_lock<std::shared_mutex> lock(metrics_mutex_);
        return counters_;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_all_counters: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get counters: " + std::string(e.what()));
    }
}

Result<std::unordered_map<std::string, double>> MetricsService::get_all_gauges() const {
    try {
        std::shared_lock<std::shared_mutex> lock(metrics_mutex_);
        return gauges_;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_all_gauges: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get gauges: " + std::string(e.what()));
    }
}

Result<std::unordered_map<std::string, HistogramData>> MetricsService::get_all_histograms() const {
    try {
        std::shared_lock<std::shared_mutex> lock(metrics_mutex_);
        return histograms_;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_all_histograms: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get histograms: " + std::string(e.what()));
    }
}

Result<bool> MetricsService::reset_metrics(const std::string& metric_name) {
    try {
        LOG_INFO(logger_, "Resetting metrics for: " + metric_name);
        
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        if (!metric_name.empty()) {
            // Reset specific metric
            metrics_.erase(metric_name);
            counters_.erase(metric_name);
            gauges_.erase(metric_name);
            histograms_.erase(metric_name);
        } else {
            // Reset all metrics
            metrics_.clear();
            counters_.clear();
            gauges_.clear();
            histograms_.clear();
        }
        
        LOG_INFO(logger_, "Metrics reset successfully for: " + 
                (metric_name.empty() ? "all metrics" : metric_name));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in reset_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to reset metrics: " + std::string(e.what()));
    }
}

Result<std::string> MetricsService::export_metrics_prometheus() const {
    try {
        std::shared_lock<std::shared_mutex> lock(metrics_mutex_);
        
        std::string prometheus_output;
        
        // Export counters
        for (const auto& counter : counters_) {
            prometheus_output += "# TYPE " + counter.first + " counter\n";
            prometheus_output += counter.first + " " + std::to_string(counter.second) + "\n\n";
        }
        
        // Export gauges
        for (const auto& gauge : gauges_) {
            prometheus_output += "# TYPE " + gauge.first + " gauge\n";
            prometheus_output += gauge.first + " " + std::to_string(gauge.second) + "\n\n";
        }
        
        // Export histograms
        for (const auto& histogram : histograms_) {
            const auto& hist_name = histogram.first;
            const auto& hist_data = histogram.second;
            
            prometheus_output += "# TYPE " + hist_name + " histogram\n";
            for (const auto& bucket : hist_data.buckets) {
                prometheus_output += hist_name + "_bucket{le=\"" + std::to_string(bucket.upper_bound) + "\"} " + 
                                   std::to_string(bucket.count) + "\n";
            }
            prometheus_output += hist_name + "_sum " + std::to_string(hist_data.sum) + "\n";
            prometheus_output += hist_name + "_count " + std::to_string(hist_data.count) + "\n\n";
        }
        
        LOG_DEBUG(logger_, "Exported Prometheus metrics, size: " + 
                 std::to_string(prometheus_output.length()) + " chars");
        return prometheus_output;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in export_metrics_prometheus: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to export Prometheus metrics: " + std::string(e.what()));
    }
}

MetricsConfig MetricsService::get_config() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

Result<bool> MetricsService::update_metrics_config(const MetricsConfig& new_config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (!validate_config(new_config)) {
            LOG_ERROR(logger_, "Invalid metrics configuration provided for update");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid metrics configuration");
        }
        
        config_ = new_config;
        
        LOG_INFO(logger_, "Updated metrics configuration: collection_interval=" + 
                std::to_string(config_.collection_interval_seconds) + "s, " +
                "retention_period=" + std::to_string(config_.retention_period_days) + " days");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_metrics_config: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update metrics configuration: " + std::string(e.what()));
    }
}

Result<std::vector<MetricData>> MetricsService::query_metrics(const std::string& metric_name,
                                                         std::chrono::milliseconds start_time,
                                                         std::chrono::milliseconds end_time) const {
    try {
        std::shared_lock<std::shared_mutex> lock(metrics_mutex_);
        
        std::vector<MetricData> filtered_metrics;
        auto it = metrics_.find(metric_name);
        if (it != metrics_.end()) {
            for (const auto& metric : it->second) {
                auto metric_time = std::chrono::milliseconds(metric.timestamp);
                if (metric_time >= start_time && metric_time <= end_time) {
                    filtered_metrics.push_back(metric);
                }
            }
        }
        
        LOG_DEBUG(logger_, "Queried metrics for " + metric_name + ": found " + 
                 std::to_string(filtered_metrics.size()) + " entries");
        return filtered_metrics;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in query_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to query metrics: " + std::string(e.what()));
    }
}

Result<double> MetricsService::get_counter_value(const std::string& counter_name) const {
    try {
        std::shared_lock<std::shared_mutex> lock(metrics_mutex_);
        
        auto it = counters_.find(counter_name);
        if (it != counters_.end()) {
            return it->second;
        }
        
        LOG_DEBUG(logger_, "Counter not found: " + counter_name);
        RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Counter not found: " + counter_name);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_counter_value: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get counter value: " + std::string(e.what()));
    }
}

Result<double> MetricsService::get_gauge_value(const std::string& gauge_name) const {
    try {
        std::shared_lock<std::shared_mutex> lock(metrics_mutex_);
        
        auto it = gauges_.find(gauge_name);
        if (it != gauges_.end()) {
            return it->second;
        }
        
        LOG_DEBUG(logger_, "Gauge not found: " + gauge_name);
        RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Gauge not found: " + gauge_name);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_gauge_value: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get gauge value: " + std::string(e.what()));
    }
}

Result<HistogramData> MetricsService::get_histogram_data(const std::string& histogram_name) const {
    try {
        std::shared_lock<std::shared_mutex> lock(metrics_mutex_);
        
        auto it = histograms_.find(histogram_name);
        if (it != histograms_.end()) {
            return it->second;
        }
        
        LOG_DEBUG(logger_, "Histogram not found: " + histogram_name);
        RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Histogram not found: " + histogram_name);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_histogram_data: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get histogram data: " + std::string(e.what()));
    }
}

// Private methods

void MetricsService::run_collection_loop() {
    LOG_INFO(logger_, "Metrics collection loop started");
    
    while (running_) {
        try {
            collect_system_metrics();
            
            // Sleep for collection interval
            std::this_thread::sleep_for(std::chrono::seconds(config_.collection_interval_seconds));
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Exception in metrics collection loop: " + std::string(e.what()));
            std::this_thread::sleep_for(std::chrono::seconds(1)); // Prevent tight loop on error
        }
    }
    
    LOG_INFO(logger_, "Metrics collection loop stopped");
}

void MetricsService::collect_system_metrics() {
    try {
        LOG_DEBUG(logger_, "Collecting system metrics");
        
        // Collect CPU usage (mock implementation)
        update_gauge("system_cpu_usage", static_cast<double>(rand() % 100), {});
        
        // Collect memory usage (mock implementation)
        update_gauge("system_memory_usage", static_cast<double>(rand() % 100), {});
        
        // Collect disk usage (mock implementation)
        update_gauge("system_disk_usage", static_cast<double>(rand() % 100), {});
        
        // Collect network I/O (mock implementation)
        update_gauge("system_network_io", static_cast<double>(rand() % 10000) / 100.0, {});
        
        LOG_DEBUG(logger_, "Collected system metrics");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in collect_system_metrics: " + std::string(e.what()));
    }
}

void MetricsService::cleanup_old_metrics(const std::string& metric_name) {
    try {
        // Remove metrics older than retention period
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        auto retention_limit = now - (config_.retention_period_days * 24 * 60 * 60 * 1000LL);
        
        auto it = metrics_.find(metric_name);
        if (it != metrics_.end()) {
            auto& metric_list = it->second;
            metric_list.erase(
                std::remove_if(metric_list.begin(), metric_list.end(), 
                             [retention_limit](const MetricData& metric) {
                                 return metric.timestamp < retention_limit;
                             }),
                metric_list.end()
            );
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in cleanup_old_metrics: " + std::string(e.what()));
    }
}

bool MetricsService::validate_config(const MetricsConfig& config) const {
    // Basic validation
    if (config.collection_interval_seconds <= 0) {
        LOG_ERROR(logger_, "Invalid collection interval: " + std::to_string(config.collection_interval_seconds));
        return false;
    }
    
    if (config.retention_period_days <= 0) {
        LOG_ERROR(logger_, "Invalid retention period: " + std::to_string(config.retention_period_days));
        return false;
    }
    
    if (config.max_metrics_per_series < 0) {
        LOG_ERROR(logger_, "Invalid max metrics per series: " + std::to_string(config.max_metrics_per_series));
        return false;
    }
    
    return true;
}

std::string MetricsService::sanitize_label_name(const std::string& label_name) const {
    std::string sanitized = label_name;
    
    // Replace invalid characters with underscores
    for (char& c : sanitized) {
        if (!(isalnum(c) || c == '_')) {
            c = '_';
        }
    }
    
    // Ensure it doesn't start with a digit
    if (!sanitized.empty() && isdigit(sanitized[0])) {
        sanitized = "_" + sanitized;
    }
    
    return sanitized;
}

std::string MetricsService::escape_label_value(const std::string& label_value) const {
    std::string escaped = label_value;
    
    // Escape backslashes
    size_t pos = 0;
    while ((pos = escaped.find("\\", pos)) != std::string::npos) {
        escaped.replace(pos, 1, "\\\\");
        pos += 2;
    }
    
    // Escape quotes
    pos = 0;
    while ((pos = escaped.find("\"", pos)) != std::string::npos) {
        escaped.replace(pos, 1, "\\\"");
        pos += 2;
    }
    
    // Escape newlines
    pos = 0;
    while ((pos = escaped.find("\n", pos)) != std::string::npos) {
        escaped.replace(pos, 1, "\\n");
        pos += 2;
    }
    
    return escaped;
}

std::string MetricsService::format_labels(const std::unordered_map<std::string, std::string>& labels) const {
    if (labels.empty()) {
        return "";
    }
    
    std::string formatted_labels = "{";
    bool first = true;
    
    for (const auto& label : labels) {
        if (!first) {
            formatted_labels += ",";
        }
        formatted_labels += sanitize_label_name(label.first) + "=\"" + escape_label_value(label.second) + "\"";
        first = false;
    }
    
    formatted_labels += "}";
    return formatted_labels;
}

void MetricsService::initialize_default_histogram_buckets(const std::string& histogram_name) {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        // Initialize histogram with default buckets if not already present
        if (histograms_.find(histogram_name) == histograms_.end()) {
            HistogramData histogram;
            
            // Default buckets: 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
            std::vector<double> default_buckets = {0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0};
            
            for (double bucket : default_buckets) {
                HistogramBucket bucket_data;
                bucket_data.upper_bound = bucket;
                bucket_data.count = 0;
                histogram.buckets.push_back(bucket_data);
            }
            
            // Add infinity bucket
            HistogramBucket inf_bucket;
            inf_bucket.upper_bound = std::numeric_limits<double>::infinity();
            inf_bucket.count = 0;
            histogram.buckets.push_back(inf_bucket);
            
            histogram.sum = 0.0;
            histogram.count = 0;
            histograms_[histogram_name] = histogram;
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in initialize_default_histogram_buckets: " + std::string(e.what()));
    }
}

} // namespace jadevectordb