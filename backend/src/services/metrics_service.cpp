#include "metrics_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <ctime>

namespace jadevectordb {

MetricsService::MetricsService() {
    logger_ = logging::LoggerManager::get_logger("MetricsService");
    service_start_time_ = std::chrono::system_clock::now();
}

bool MetricsService::initialize(const MetricsConfig& config) {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        if (config.collection_interval_seconds <= 0) {
            LOG_ERROR(logger_, "Invalid collection interval: " + std::to_string(config.collection_interval_seconds));
            return false;
        }

        if (config.retention_hours <= 0) {
            LOG_ERROR(logger_, "Invalid retention hours: " + std::to_string(config.retention_hours));
            return false;
        }

        config_ = config;

        LOG_INFO(logger_, "MetricsService initialized with collection_interval: " +
                std::to_string(config_.collection_interval_seconds) + "s, " +
                "retention_hours: " + std::to_string(config_.retention_hours) + "h");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in MetricsService::initialize: " + std::string(e.what()));
        return false;
    }
}

Result<bool> MetricsService::record_metric(const std::string& name,
                                           double value,
                                           const std::unordered_map<std::string, std::string>& labels) {
    try {
        if (!is_valid_metric_name(name)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid metric name: " + name);
        }

        if (!are_valid_labels(labels)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid metric labels");
        }

        std::lock_guard<std::mutex> lock(metrics_mutex_);

        MetricRecord record;
        record.name = name;
        record.description = "Recorded metric";
        record.type = MetricType::GAUGE;
        record.value = value;
        record.timestamp = std::chrono::system_clock::now();

        // Convert unordered_map to map for labels
        for (const auto& label : labels) {
            record.labels[label.first] = label.second;
        }

        // Update current metrics
        std::string key = create_metric_key(name, labels);
        current_metrics_[key] = record;

        // Add to history
        metrics_history_.push_back(record);

        LOG_DEBUG(logger_, "Recorded metric: " + name + " = " + std::to_string(value));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in record_metric: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to record metric: " + std::string(e.what()));
    }
}

Result<bool> MetricsService::increment_counter(const std::string& name,
                                               double increment,
                                               const std::unordered_map<std::string, std::string>& labels) {
    try {
        if (!is_valid_metric_name(name)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid metric name: " + name);
        }

        std::lock_guard<std::mutex> lock(metrics_mutex_);

        std::string key = create_metric_key(name, labels);

        // Get or create counter
        if (current_metrics_.find(key) == current_metrics_.end()) {
            MetricRecord record;
            record.name = name;
            record.description = "Counter metric";
            record.type = MetricType::COUNTER;
            record.value = 0.0;
            record.timestamp = std::chrono::system_clock::now();
            for (const auto& label : labels) {
                record.labels[label.first] = label.second;
            }
            current_metrics_[key] = record;
        }

        // Increment
        current_metrics_[key].value += increment;
        current_metrics_[key].timestamp = std::chrono::system_clock::now();

        // Add to history
        metrics_history_.push_back(current_metrics_[key]);

        LOG_DEBUG(logger_, "Incremented counter: " + name + " by " + std::to_string(increment));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in increment_counter: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to increment counter: " + std::string(e.what()));
    }
}

Result<bool> MetricsService::set_gauge(const std::string& name,
                                       double value,
                                       const std::unordered_map<std::string, std::string>& labels) {
    try {
        if (!is_valid_metric_name(name)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid metric name: " + name);
        }

        std::lock_guard<std::mutex> lock(metrics_mutex_);

        MetricRecord record;
        record.name = name;
        record.description = "Gauge metric";
        record.type = MetricType::GAUGE;
        record.value = value;
        record.timestamp = std::chrono::system_clock::now();
        for (const auto& label : labels) {
            record.labels[label.first] = label.second;
        }

        std::string key = create_metric_key(name, labels);
        current_metrics_[key] = record;

        // Add to history
        metrics_history_.push_back(record);

        LOG_DEBUG(logger_, "Set gauge: " + name + " = " + std::to_string(value));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in set_gauge: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to set gauge: " + std::string(e.what()));
    }
}

Result<bool> MetricsService::record_histogram(const std::string& name,
                                              double value,
                                              const std::unordered_map<std::string, std::string>& labels) {
    try {
        if (!is_valid_metric_name(name)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid metric name: " + name);
        }

        std::lock_guard<std::mutex> lock(metrics_mutex_);

        MetricRecord record;
        record.name = name;
        record.description = "Histogram metric";
        record.type = MetricType::HISTOGRAM;
        record.value = value;
        record.timestamp = std::chrono::system_clock::now();
        for (const auto& label : labels) {
            record.labels[label.first] = label.second;
        }

        std::string key = create_metric_key(name, labels);
        current_metrics_[key] = record;

        // Add to history
        metrics_history_.push_back(record);

        LOG_DEBUG(logger_, "Recorded histogram: " + name + " = " + std::to_string(value));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in record_histogram: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to record histogram: " + std::string(e.what()));
    }
}

Result<double> MetricsService::get_metric_value(const std::string& name) const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        // Find first metric with matching name
        for (const auto& pair : current_metrics_) {
            if (pair.second.name == name) {
                return pair.second.value;
            }
        }

        RETURN_ERROR(ErrorCode::NOT_FOUND, "Metric not found: " + name);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_metric_value: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get metric value: " + std::string(e.what()));
    }
}

Result<MetricRecord> MetricsService::get_metric(const std::string& name) const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        // Find first metric with matching name
        for (const auto& pair : current_metrics_) {
            if (pair.second.name == name) {
                return pair.second;
            }
        }

        RETURN_ERROR(ErrorCode::NOT_FOUND, "Metric not found: " + name);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_metric: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get metric: " + std::string(e.what()));
    }
}

Result<std::vector<MetricRecord>> MetricsService::get_all_metrics() const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        std::vector<MetricRecord> metrics;
        for (const auto& pair : current_metrics_) {
            metrics.push_back(pair.second);
        }

        LOG_DEBUG(logger_, "Retrieved " + std::to_string(metrics.size()) + " metrics");
        return metrics;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_all_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get all metrics: " + std::string(e.what()));
    }
}

Result<std::vector<MetricRecord>> MetricsService::get_metrics_by_type(MetricType type) const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        std::vector<MetricRecord> filtered;
        for (const auto& pair : current_metrics_) {
            if (pair.second.type == type) {
                filtered.push_back(pair.second);
            }
        }

        LOG_DEBUG(logger_, "Retrieved " + std::to_string(filtered.size()) + " metrics of specified type");
        return filtered;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_metrics_by_type: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get metrics by type: " + std::string(e.what()));
    }
}

Result<std::vector<MetricRecord>> MetricsService::get_metrics_by_label(const std::string& label_key,
                                                                       const std::string& label_value) const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        std::vector<MetricRecord> filtered;
        for (const auto& pair : current_metrics_) {
            auto it = pair.second.labels.find(label_key);
            if (it != pair.second.labels.end() && it->second == label_value) {
                filtered.push_back(pair.second);
            }
        }

        LOG_DEBUG(logger_, "Retrieved " + std::to_string(filtered.size()) + " metrics with label " +
                 label_key + "=" + label_value);
        return filtered;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_metrics_by_label: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get metrics by label: " + std::string(e.what()));
    }
}

Result<std::vector<MetricRecord>> MetricsService::get_metrics_by_name_pattern(const std::string& pattern) const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        std::vector<MetricRecord> filtered;
        for (const auto& pair : current_metrics_) {
            // Simple wildcard matching: * matches any characters
            if (pattern.find('*') != std::string::npos) {
                std::string prefix = pattern.substr(0, pattern.find('*'));
                if (pair.second.name.find(prefix) == 0) {
                    filtered.push_back(pair.second);
                }
            } else {
                // Exact match
                if (pair.second.name == pattern) {
                    filtered.push_back(pair.second);
                }
            }
        }

        LOG_DEBUG(logger_, "Retrieved " + std::to_string(filtered.size()) + " metrics matching pattern: " + pattern);
        return filtered;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_metrics_by_name_pattern: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get metrics by pattern: " + std::string(e.what()));
    }
}

Result<std::string> MetricsService::export_metrics() const {
    try {
        if (config_.export_format == "prometheus") {
            return export_prometheus_format();
        } else if (config_.export_format == "json") {
            return export_json_format();
        } else {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Unsupported export format: " + config_.export_format);
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in export_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to export metrics: " + std::string(e.what()));
    }
}

Result<std::string> MetricsService::export_prometheus_format() const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        std::stringstream ss;

        for (const auto& pair : current_metrics_) {
            const auto& metric = pair.second;

            // Type declaration
            std::string type_str;
            switch (metric.type) {
                case MetricType::COUNTER: type_str = "counter"; break;
                case MetricType::GAUGE: type_str = "gauge"; break;
                case MetricType::HISTOGRAM: type_str = "histogram"; break;
                default: type_str = "untyped"; break;
            }

            ss << "# TYPE " << metric.name << " " << type_str << "\n";

            // Metric value with labels
            ss << metric.name;
            if (!metric.labels.empty()) {
                ss << "{";
                bool first = true;
                for (const auto& label : metric.labels) {
                    if (!first) ss << ",";
                    ss << label.first << "=\"" << label.second << "\"";
                    first = false;
                }
                ss << "}";
            }
            ss << " " << metric.value << "\n";
        }

        LOG_DEBUG(logger_, "Exported metrics in Prometheus format");
        return ss.str();
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in export_prometheus_format: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to export Prometheus format: " + std::string(e.what()));
    }
}

Result<std::string> MetricsService::export_json_format() const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        std::stringstream ss;
        ss << "{\n  \"metrics\": [\n";

        bool first = true;
        for (const auto& pair : current_metrics_) {
            const auto& metric = pair.second;

            if (!first) ss << ",\n";

            ss << "    {\n";
            ss << "      \"name\": \"" << metric.name << "\",\n";
            ss << "      \"description\": \"" << metric.description << "\",\n";
            ss << "      \"type\": \"";
            switch (metric.type) {
                case MetricType::COUNTER: ss << "counter"; break;
                case MetricType::GAUGE: ss << "gauge"; break;
                case MetricType::HISTOGRAM: ss << "histogram"; break;
                default: ss << "untyped"; break;
            }
            ss << "\",\n";
            ss << "      \"value\": " << metric.value << ",\n";
            ss << "      \"timestamp\": \"" << format_timestamp(metric.timestamp) << "\",\n";
            ss << "      \"labels\": {";

            bool first_label = true;
            for (const auto& label : metric.labels) {
                if (!first_label) ss << ", ";
                ss << "\"" << label.first << "\": \"" << label.second << "\"";
                first_label = false;
            }
            ss << "}\n";
            ss << "    }";

            first = false;
        }

        ss << "\n  ]\n}";

        LOG_DEBUG(logger_, "Exported metrics in JSON format");
        return ss.str();
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in export_json_format: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to export JSON format: " + std::string(e.what()));
    }
}

Result<std::vector<MetricRecord>> MetricsService::get_metrics_history_minutes(int minutes) const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        auto cutoff_time = std::chrono::system_clock::now() - std::chrono::minutes(minutes);

        std::vector<MetricRecord> filtered;
        for (const auto& metric : metrics_history_) {
            if (metric.timestamp >= cutoff_time) {
                filtered.push_back(metric);
            }
        }

        LOG_DEBUG(logger_, "Retrieved " + std::to_string(filtered.size()) + " metrics from last " +
                 std::to_string(minutes) + " minutes");
        return filtered;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_metrics_history_minutes: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get metrics history: " + std::string(e.what()));
    }
}

Result<std::vector<MetricRecord>> MetricsService::get_metrics_history_hours(int hours) const {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        auto cutoff_time = std::chrono::system_clock::now() - std::chrono::hours(hours);

        std::vector<MetricRecord> filtered;
        for (const auto& metric : metrics_history_) {
            if (metric.timestamp >= cutoff_time) {
                filtered.push_back(metric);
            }
        }

        LOG_DEBUG(logger_, "Retrieved " + std::to_string(filtered.size()) + " metrics from last " +
                 std::to_string(hours) + " hours");
        return filtered;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_metrics_history_hours: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get metrics history: " + std::string(e.what()));
    }
}

Result<bool> MetricsService::cleanup_old_metrics() {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        auto cutoff_time = std::chrono::system_clock::now() - std::chrono::hours(config_.retention_hours);

        size_t original_size = metrics_history_.size();

        metrics_history_.erase(
            std::remove_if(metrics_history_.begin(), metrics_history_.end(),
                [cutoff_time](const MetricRecord& metric) {
                    return metric.timestamp < cutoff_time;
                }),
            metrics_history_.end()
        );

        size_t removed = original_size - metrics_history_.size();

        LOG_INFO(logger_, "Cleaned up " + std::to_string(removed) + " old metrics");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in cleanup_old_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to cleanup metrics: " + std::string(e.what()));
    }
}

Result<bool> MetricsService::update_config(const MetricsConfig& new_config) {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        if (new_config.collection_interval_seconds <= 0) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid collection interval");
        }

        if (new_config.retention_hours <= 0) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid retention hours");
        }

        config_ = new_config;

        LOG_INFO(logger_, "Updated metrics configuration");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_config: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update config: " + std::string(e.what()));
    }
}

MetricsConfig MetricsService::get_config() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return config_;
}

Result<bool> MetricsService::reset_metrics() {
    try {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        current_metrics_.clear();
        metrics_history_.clear();

        LOG_INFO(logger_, "Reset all metrics");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in reset_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to reset metrics: " + std::string(e.what()));
    }
}

// Private helper methods

std::string MetricsService::create_metric_key(const std::string& name,
                                              const std::unordered_map<std::string, std::string>& labels) const {
    std::stringstream ss;
    ss << name;

    if (!labels.empty()) {
        ss << "{";
        bool first = true;
        for (const auto& label : labels) {
            if (!first) ss << ",";
            ss << label.first << "=" << label.second;
            first = false;
        }
        ss << "}";
    }

    return ss.str();
}

bool MetricsService::is_valid_metric_name(const std::string& name) const {
    if (name.empty()) {
        return false;
    }

    // Check first character is letter or underscore
    if (!std::isalpha(name[0]) && name[0] != '_') {
        return false;
    }

    // Check remaining characters are alphanumeric or underscore
    for (char c : name) {
        if (!std::isalnum(c) && c != '_') {
            return false;
        }
    }

    return true;
}

bool MetricsService::are_valid_labels(const std::unordered_map<std::string, std::string>& labels) const {
    for (const auto& label : labels) {
        if (label.first.empty() || label.second.empty()) {
            return false;
        }
    }
    return true;
}

bool MetricsService::is_valid_metric_type(MetricType type) const {
    return type == MetricType::COUNTER ||
           type == MetricType::GAUGE ||
           type == MetricType::HISTOGRAM;
}

void MetricsService::apply_retention_policy() {
    auto result = cleanup_old_metrics();
    if (!result.has_value()) {
        LOG_WARN(logger_, "Failed to apply retention policy");
    }
}

std::string MetricsService::format_timestamp(const std::chrono::system_clock::time_point& time) const {
    auto time_t = std::chrono::system_clock::to_time_t(time);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

std::unordered_map<double, double> MetricsService::calculate_percentiles(const std::vector<double>& values) const {
    std::unordered_map<double, double> percentiles;

    if (values.empty()) {
        return percentiles;
    }

    std::vector<double> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());

    // Calculate common percentiles
    std::vector<double> percentile_points = {0.5, 0.90, 0.95, 0.99};

    for (double p : percentile_points) {
        size_t index = static_cast<size_t>(p * sorted_values.size());
        if (index >= sorted_values.size()) {
            index = sorted_values.size() - 1;
        }
        percentiles[p] = sorted_values[index];
    }

    return percentiles;
}

void MetricsService::set_service_start_time() {
    service_start_time_ = std::chrono::system_clock::now();
}

Result<std::vector<MetricRecord>> MetricsService::generate_system_metrics() const {
    try {
        std::vector<MetricRecord> system_metrics;

        // Uptime metric
        auto now = std::chrono::system_clock::now();
        auto uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - service_start_time_).count();

        MetricRecord uptime_metric;
        uptime_metric.name = "system_uptime_seconds";
        uptime_metric.description = "System uptime in seconds";
        uptime_metric.type = MetricType::COUNTER;
        uptime_metric.value = static_cast<double>(uptime_seconds);
        uptime_metric.timestamp = now;
        system_metrics.push_back(uptime_metric);

        return system_metrics;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in generate_system_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to generate system metrics: " + std::string(e.what()));
    }
}

Result<std::vector<MetricRecord>> MetricsService::generate_application_metrics() const {
    try {
        std::vector<MetricRecord> app_metrics;

        // Total metrics count
        MetricRecord total_metrics;
        total_metrics.name = "app_metrics_total";
        total_metrics.description = "Total number of metrics tracked";
        total_metrics.type = MetricType::GAUGE;
        total_metrics.value = static_cast<double>(current_metrics_.size());
        total_metrics.timestamp = std::chrono::system_clock::now();
        app_metrics.push_back(total_metrics);

        return app_metrics;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in generate_application_metrics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to generate application metrics: " + std::string(e.what()));
    }
}

} // namespace jadevectordb
