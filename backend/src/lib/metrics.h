#ifndef JADEVECTORDB_METRICS_H
#define JADEVECTORDB_METRICS_H

#include <string>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <functional>
#include <atomic>
#include <vector>
#include <sstream>
#include <iomanip>

#include "logging.h"

namespace jadevectordb {

// Enum for different metric types
enum class MetricType {
    COUNTER,
    GAUGE,
    HISTOGRAM,
    SUMMARY
};

// Base class for all metrics
class Metric {
protected:
    std::string name_;
    std::string description_;
    std::map<std::string, std::string> labels_;
    std::chrono::system_clock::time_point created_at_;

public:
    Metric(const std::string& name, const std::string& description, 
           const std::map<std::string, std::string>& labels = {});
    virtual ~Metric() = default;
    
    virtual MetricType get_type() const = 0;
    virtual std::string to_prometheus_format() const = 0;
    
    const std::string& get_name() const { return name_; }
    const std::string& get_description() const { return description_; }
    const std::map<std::string, std::string>& get_labels() const { return labels_; }
    std::chrono::system_clock::time_point get_created_at() const { return created_at_; }
};

// Counter metric - monotonically increasing value
class Counter : public Metric {
private:
    mutable std::mutex mutex_;
    std::atomic<double> value_;

public:
    Counter(const std::string& name, const std::string& description,
            const std::map<std::string, std::string>& labels = {});
    
    void increment(double amount = 1.0);
    void add(double amount);
    double get_value() const;
    
    MetricType get_type() const override { return MetricType::COUNTER; }
    std::string to_prometheus_format() const override;
};

// Gauge metric - can go up and down
class Gauge : public Metric {
private:
    mutable std::mutex mutex_;
    std::atomic<double> value_;

public:
    Gauge(const std::string& name, const std::string& description,
          const std::map<std::string, std::string>& labels = {});
    
    void set(double value);
    void increment(double amount = 1.0);
    void decrement(double amount = 1.0);
    double get_value() const;
    
    MetricType get_type() const override { return MetricType::GAUGE; }
    std::string to_prometheus_format() const override;
};

// Histogram metric - samples observations and counts them in buckets
class Histogram : public Metric {
private:
    mutable std::mutex mutex_;
    std::vector<double> buckets_;
    std::vector<uint64_t> bucket_counts_;
    std::atomic<double> sum_;
    std::atomic<uint64_t> count_;

public:
    Histogram(const std::string& name, const std::string& description,
              const std::vector<double>& buckets,
              const std::map<std::string, std::string>& labels = {});
    
    void observe(double value);
    std::pair<double, uint64_t> get_sum_and_count() const;
    std::vector<std::pair<double, uint64_t>> get_buckets() const;
    
    MetricType get_type() const override { return MetricType::HISTOGRAM; }
    std::string to_prometheus_format() const override;
};

// Summary metric - similar to histogram but calculates quantiles over a sliding time window
class Summary : public Metric {
private:
    mutable std::mutex mutex_;
    std::vector<double> quantiles_; // e.g., {0.5, 0.9, 0.99} for 50th, 90th, 99th percentiles
    std::vector<double> values_;
    std::chrono::system_clock::time_point last_updated_;
    std::chrono::milliseconds max_age_;
    int max_size_;
    std::atomic<double> sum_;
    std::atomic<uint64_t> count_;

public:
    Summary(const std::string& name, const std::string& description,
            const std::vector<double>& quantiles,
            std::chrono::milliseconds max_age = std::chrono::milliseconds(60000), // 1 minute
            int max_size = 1000,
            const std::map<std::string, std::string>& labels = {});
    
    void observe(double value);
    std::map<double, double> get_quantiles() const; // Returns quantile -> value map
    std::pair<double, uint64_t> get_sum_and_count() const;
    
    MetricType get_type() const override { return MetricType::SUMMARY; }
    std::string to_prometheus_format() const override;
};

// MetricsRegistry to manage all metrics
class MetricsRegistry {
private:
    mutable std::shared_mutex registry_mutex_;
    std::map<std::string, std::shared_ptr<Metric>> metrics_;
    std::shared_ptr<logging::Logger> logger_;

public:
    MetricsRegistry();
    ~MetricsRegistry() = default;
    
    // Register different types of metrics
    std::shared_ptr<Counter> register_counter(const std::string& name, 
                                            const std::string& description,
                                            const std::map<std::string, std::string>& labels = {});
    
    std::shared_ptr<Gauge> register_gauge(const std::string& name,
                                        const std::string& description,
                                        const std::map<std::string, std::string>& labels = {});
    
    std::shared_ptr<Histogram> register_histogram(const std::string& name,
                                                const std::string& description,
                                                const std::vector<double>& buckets,
                                                const std::map<std::string, std::string>& labels = {});
    
    std::shared_ptr<Summary> register_summary(const std::string& name,
                                            const std::string& description,
                                            const std::vector<double>& quantiles,
                                            std::chrono::milliseconds max_age = std::chrono::milliseconds(60000),
                                            int max_size = 1000,
                                            const std::map<std::string, std::string>& labels = {});
    
    // Remove a metric
    void unregister_metric(const std::string& name);
    
    // Get all metrics in Prometheus format
    std::string to_prometheus_format() const;
    
    // Get a specific metric
    std::shared_ptr<Metric> get_metric(const std::string& name) const;
    
    // Get all metric names
    std::vector<std::string> get_metric_names() const;
    
    // Reset all metrics
    void reset();
};

// Global metrics registry singleton
class MetricsManager {
private:
    static std::unique_ptr<MetricsRegistry> registry_;
    static std::once_flag once_flag_;

public:
    static MetricsRegistry* get_registry();
    
private:
    MetricsManager() = default;
    ~MetricsManager() = default;
    MetricsManager(const MetricsManager&) = delete;
    MetricsManager& operator=(const MetricsManager&) = delete;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_METRICS_H