#include "metrics.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace jadevectordb {

// Metric implementation
Metric::Metric(const std::string& name, const std::string& description,
               const std::map<std::string, std::string>& labels)
    : name_(name), description_(description), labels_(labels),
      created_at_(std::chrono::system_clock::now()) {
}

// Counter implementation
Counter::Counter(const std::string& name, const std::string& description,
                 const std::map<std::string, std::string>& labels)
    : Metric(name, description, labels), value_(0.0) {
}

void Counter::increment(double amount) {
    if (amount < 0) {
        // Counters should only increase
        return;
    }
    value_.fetch_add(amount, std::memory_order_relaxed);
}

void Counter::add(double amount) {
    if (amount < 0) {
        // Counters should only increase
        return;
    }
    value_.fetch_add(amount, std::memory_order_relaxed);
}

double Counter::get_value() const {
    return value_.load(std::memory_order_relaxed);
}

std::string Counter::to_prometheus_format() const {
    std::ostringstream oss;
    
    // Add HELP and TYPE info
    oss << "# HELP " << name_ << " " << description_ << "\n";
    oss << "# TYPE " << name_ << " counter\n";
    
    // Add the metric value
    oss << name_;
    
    // Add labels if any
    if (!labels_.empty()) {
        oss << "{";
        bool first = true;
        for (const auto& [key, value] : labels_) {
            if (!first) oss << ",";
            oss << key << "=\"" << value << "\"";
            first = false;
        }
        oss << "}";
    }
    
    oss << " " << std::fixed << std::setprecision(2) << get_value() << "\n";
    
    return oss.str();
}

// Gauge implementation
Gauge::Gauge(const std::string& name, const std::string& description,
             const std::map<std::string, std::string>& labels)
    : Metric(name, description, labels), value_(0.0) {
}

void Gauge::set(double value) {
    value_.store(value, std::memory_order_relaxed);
}

void Gauge::increment(double amount) {
    value_.fetch_add(amount, std::memory_order_relaxed);
}

void Gauge::decrement(double amount) {
    value_.fetch_sub(amount, std::memory_order_relaxed);
}

double Gauge::get_value() const {
    return value_.load(std::memory_order_relaxed);
}

std::string Gauge::to_prometheus_format() const {
    std::ostringstream oss;
    
    // Add HELP and TYPE info
    oss << "# HELP " << name_ << " " << description_ << "\n";
    oss << "# TYPE " << name_ << " gauge\n";
    
    // Add the metric value
    oss << name_;
    
    // Add labels if any
    if (!labels_.empty()) {
        oss << "{";
        bool first = true;
        for (const auto& [key, value] : labels_) {
            if (!first) oss << ",";
            oss << key << "=\"" << value << "\"";
            first = false;
        }
        oss << "}";
    }
    
    oss << " " << std::fixed << std::setprecision(2) << get_value() << "\n";
    
    return oss.str();
}

// Histogram implementation
Histogram::Histogram(const std::string& name, const std::string& description,
                     const std::vector<double>& buckets,
                     const std::map<std::string, std::string>& labels)
    : Metric(name, description, labels), buckets_(buckets), sum_(0.0), count_(0) {
    // Initialize bucket counts to 0
    bucket_counts_.resize(buckets_.size() + 1, 0); // +1 for the +Inf bucket
}

void Histogram::observe(double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Update sum and count
    sum_.fetch_add(value, std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
    
    // Determine which bucket this value falls into
    for (size_t i = 0; i < buckets_.size(); ++i) {
        if (value <= buckets_[i]) {
            ++bucket_counts_[i];
            return;
        }
    }
    
    // If the value is greater than all buckets, put it in the +Inf bucket
    ++bucket_counts_.back();
}

std::pair<double, uint64_t> Histogram::get_sum_and_count() const {
    return {sum_.load(std::memory_order_relaxed), count_.load(std::memory_order_relaxed)};
}

std::vector<std::pair<double, uint64_t>> Histogram::get_buckets() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::pair<double, uint64_t>> result;
    
    // Add counts for each defined bucket
    for (size_t i = 0; i < buckets_.size(); ++i) {
        result.emplace_back(buckets_[i], bucket_counts_[i]);
    }
    
    // Add the +Inf bucket
    result.emplace_back(std::numeric_limits<double>::infinity(), 
                        bucket_counts_.back());
    
    return result;
}

std::string Histogram::to_prometheus_format() const {
    std::ostringstream oss;
    
    // Add HELP and TYPE info
    oss << "# HELP " << name_ << " " << description_ << "\n";
    oss << "# TYPE " << name_ << " histogram\n";
    
    auto buckets = get_buckets();
    auto sum_and_count = get_sum_and_count();
    
    // Add buckets
    for (size_t i = 0; i < buckets.size(); ++i) {
        oss << name_ << "_bucket{";
        
        // Add labels if any
        bool first = true;
        for (const auto& [key, value] : labels_) {
            if (!first) oss << ",";
            oss << key << "=\"" << value << "\"";
            first = false;
        }
        
        if (!first) oss << ",";
        oss << "le=\"" << std::fixed << std::setprecision(2) << buckets[i].first << "\"} ";
        oss << buckets[i].second << "\n";
    }
    
    // Add count
    oss << name_ << "_count";
    if (!labels_.empty()) {
        oss << "{";
        bool first = true;
        for (const auto& [key, value] : labels_) {
            if (!first) oss << ",";
            oss << key << "=\"" << value << "\"";
            first = false;
        }
        oss << "}";
    }
    oss << " " << sum_and_count.second << "\n";
    
    // Add sum
    oss << name_ << "_sum";
    if (!labels_.empty()) {
        oss << "{";
        bool first = true;
        for (const auto& [key, value] : labels_) {
            if (!first) oss << ",";
            oss << key << "=\"" << value << "\"";
            first = false;
        }
        oss << "}";
    }
    oss << " " << std::fixed << std::setprecision(2) << sum_and_count.first << "\n";
    
    return oss.str();
}

// Summary implementation
Summary::Summary(const std::string& name, const std::string& description,
                 const std::vector<double>& quantiles,
                 std::chrono::milliseconds max_age, int max_size,
                 const std::map<std::string, std::string>& labels)
    : Metric(name, description, labels), quantiles_(quantiles), max_age_(max_age), 
      max_size_(max_size), sum_(0.0), count_(0) {
    last_updated_ = std::chrono::system_clock::now();
}

void Summary::observe(double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Clean up old values based on max_age_
    auto now = std::chrono::system_clock::now();
    if (now - last_updated_ > max_age_) {
        // For a basic implementation, we'll just clear old values periodically
        values_.clear();
        last_updated_ = now;
    }
    
    // Add value to the list
    values_.push_back(value);
    
    // Keep only the most recent values up to max_size_
    if (values_.size() > max_size_) {
        values_.erase(values_.begin(), values_.begin() + (values_.size() - max_size_));
    }
    
    // Update sum and count
    sum_.fetch_add(value, std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
}

std::map<double, double> Summary::get_quantiles() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (values_.empty()) {
        return {}; // Return empty map if no values
    }
    
    // Copy the values and sort them for quantile calculation
    std::vector<double> sorted_values = values_;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    std::map<double, double> result;
    
    for (double quantile : quantiles_) {
        if (quantile < 0 || quantile > 1.0) continue; // Skip invalid quantiles
        
        // Calculate index for the quantile
        size_t index = static_cast<size_t>(quantile * (sorted_values.size() - 1));
        
        if (index < sorted_values.size()) {
            result[quantile] = sorted_values[index];
        }
    }
    
    return result;
}

std::pair<double, uint64_t> Summary::get_sum_and_count() const {
    return {sum_.load(std::memory_order_relaxed), count_.load(std::memory_order_relaxed)};
}

std::string Summary::to_prometheus_format() const {
    std::ostringstream oss;
    
    // Add HELP and TYPE info
    oss << "# HELP " << name_ << " " << description_ << "\n";
    oss << "# TYPE " << name_ << " summary\n";
    
    auto quantiles = get_quantiles();
    auto sum_and_count = get_sum_and_count();
    
    // Add quantiles
    for (const auto& pair : quantiles) {
        oss << name_ << "{";
        
        // Add labels if any
        bool first = true;
        for (const auto& [key, value] : labels_) {
            if (!first) oss << ",";
            oss << key << "=\"" << value << "\"";
            first = false;
        }
        
        if (!first) oss << ",";
        oss << "quantile=\"" << std::fixed << std::setprecision(2) << pair.first << "\"} ";
        oss << std::fixed << std::setprecision(2) << pair.second << "\n";
    }
    
    // Add count
    oss << name_ << "_count";
    if (!labels_.empty()) {
        oss << "{";
        bool first = true;
        for (const auto& [key, value] : labels_) {
            if (!first) oss << ",";
            oss << key << "=\"" << value << "\"";
            first = false;
        }
        oss << "}";
    }
    oss << " " << sum_and_count.second << "\n";
    
    // Add sum
    oss << name_ << "_sum";
    if (!labels_.empty()) {
        oss << "{";
        bool first = true;
        for (const auto& [key, value] : labels_) {
            if (!first) oss << ",";
            oss << key << "=\"" << value << "\"";
            first = false;
        }
        oss << "}";
    }
    oss << " " << std::fixed << std::setprecision(2) << sum_and_count.first << "\n";
    
    return oss.str();
}



MetricsRegistry::MetricsRegistry() {
    logger_ = logging::LoggerManager::get_logger("MetricsRegistry");
}

std::shared_ptr<Counter> MetricsRegistry::register_counter(
    const std::string& name, 
    const std::string& description,
    const std::map<std::string, std::string>& labels) {
    
    std::unique_lock<std::shared_mutex> lock(registry_mutex_);
    
    // Check if metric already exists
    if (metrics_.find(name) != metrics_.end()) {
        // In a real implementation, we might want to return the existing metric
        return nullptr;
    }
    
    auto counter = std::make_shared<Counter>(name, description, labels);
    metrics_[name] = counter;
    
    return counter;
}

std::shared_ptr<Gauge> MetricsRegistry::register_gauge(
    const std::string& name,
    const std::string& description,
    const std::map<std::string, std::string>& labels) {
    
    std::unique_lock<std::shared_mutex> lock(registry_mutex_);
    
    // Check if metric already exists
    if (metrics_.find(name) != metrics_.end()) {
        return nullptr;
    }
    
    auto gauge = std::make_shared<Gauge>(name, description, labels);
    metrics_[name] = gauge;
    
    return gauge;
}

std::shared_ptr<Histogram> MetricsRegistry::register_histogram(
    const std::string& name,
    const std::string& description,
    const std::vector<double>& buckets,
    const std::map<std::string, std::string>& labels) {
    
    std::unique_lock<std::shared_mutex> lock(registry_mutex_);
    
    // Check if metric already exists
    if (metrics_.find(name) != metrics_.end()) {
        return nullptr;
    }
    
    auto histogram = std::make_shared<Histogram>(name, description, buckets, labels);
    metrics_[name] = histogram;
    
    return histogram;
}

std::shared_ptr<Summary> MetricsRegistry::register_summary(
    const std::string& name,
    const std::string& description,
    const std::vector<double>& quantiles,
    std::chrono::milliseconds max_age,
    int max_size,
    const std::map<std::string, std::string>& labels) {
    
    std::unique_lock<std::shared_mutex> lock(registry_mutex_);
    
    // Check if metric already exists
    if (metrics_.find(name) != metrics_.end()) {
        return nullptr;
    }
    
    auto summary = std::make_shared<Summary>(name, description, quantiles, max_age, max_size, labels);
    metrics_[name] = summary;
    
    return summary;
}

void MetricsRegistry::unregister_metric(const std::string& name) {
    std::unique_lock<std::shared_mutex> lock(registry_mutex_);
    metrics_.erase(name);
}

std::string MetricsRegistry::to_prometheus_format() const {
    std::shared_lock<std::shared_mutex> lock(registry_mutex_);
    
    std::ostringstream oss;
    
    for (const auto& [name, metric] : metrics_) {
        oss << metric->to_prometheus_format();
    }
    
    return oss.str();
}

std::shared_ptr<Metric> MetricsRegistry::get_metric(const std::string& name) const {
    std::shared_lock<std::shared_mutex> lock(registry_mutex_);
    
    auto it = metrics_.find(name);
    if (it != metrics_.end()) {
        return it->second;
    }
    
    return nullptr;
}

std::vector<std::string> MetricsRegistry::get_metric_names() const {
    std::shared_lock<std::shared_mutex> lock(registry_mutex_);
    
    std::vector<std::string> names;
    for (const auto& [name, metric] : metrics_) {
        names.push_back(name);
    }
    
    return names;
}

void MetricsRegistry::reset() {
    std::unique_lock<std::shared_mutex> lock(registry_mutex_);
    metrics_.clear();
}

// MetricsManager implementation
std::unique_ptr<MetricsRegistry> MetricsManager::registry_ = nullptr;
std::once_flag MetricsManager::once_flag_;

MetricsRegistry* MetricsManager::get_registry() {
    std::call_once(once_flag_, []() {
        registry_ = std::unique_ptr<MetricsRegistry>(new MetricsRegistry());
    });
    return registry_.get();
}

} // namespace jadevectordb