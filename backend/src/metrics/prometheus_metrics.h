#ifndef JADEVECTORDB_PROMETHEUS_METRICS_H
#define JADEVECTORDB_PROMETHEUS_METRICS_H

#include "lib/metrics.h"
#include <memory>
#include <string>
#include <chrono>

namespace jadevectordb {

/**
 * @brief Prometheus metrics manager for JadeVectorDB
 * 
 * Centralizes all application metrics for export to Prometheus.
 * Provides convenient methods to record metrics across all services.
 */
class PrometheusMetrics {
private:
    MetricsRegistry* registry_;
    
    // Authentication metrics
    std::shared_ptr<Counter> auth_requests_total_;
    std::shared_ptr<Histogram> auth_duration_seconds_;
    std::shared_ptr<Counter> auth_errors_total_;
    std::shared_ptr<Gauge> active_sessions_;
    
    // Permission check metrics
    std::shared_ptr<Counter> permission_checks_total_;
    std::shared_ptr<Histogram> permission_check_duration_seconds_;
    std::shared_ptr<Counter> permission_cache_hits_total_;
    std::shared_ptr<Counter> permission_cache_misses_total_;
    
    // Database operation metrics
    std::shared_ptr<Counter> db_operations_total_;
    std::shared_ptr<Histogram> db_operation_duration_seconds_;
    std::shared_ptr<Counter> db_connection_errors_total_;
    std::shared_ptr<Counter> db_query_retries_total_;
    
    // User management metrics
    std::shared_ptr<Gauge> users_total_;
    std::shared_ptr<Counter> user_operations_total_;
    std::shared_ptr<Counter> failed_logins_total_;
    std::shared_ptr<Gauge> locked_accounts_total_;
    
    // Rate limiting metrics
    std::shared_ptr<Counter> rate_limit_exceeded_total_;
    std::shared_ptr<Counter> ip_blocks_total_;
    
    // Circuit breaker metrics
    std::shared_ptr<Counter> circuit_breaker_state_changes_total_;
    std::shared_ptr<Gauge> circuit_breaker_open_;

public:
    PrometheusMetrics();
    ~PrometheusMetrics() = default;
    
    // Initialize all metrics
    void initialize();
    
    // Get metrics in Prometheus format
    std::string get_metrics() const;
    
    // Authentication metrics recording
    void record_auth_request(const std::string& method, const std::string& status);
    void record_auth_duration(const std::string& method, double duration_seconds);
    void record_auth_error(const std::string& method, const std::string& error_type);
    void set_active_sessions(int count);
    void increment_active_sessions();
    void decrement_active_sessions();
    
    // Permission check metrics recording
    void record_permission_check(const std::string& permission, const std::string& result);
    void record_permission_check_duration(double duration_seconds);
    void record_permission_cache_hit();
    void record_permission_cache_miss();
    
    // Database operation metrics recording
    void record_db_operation(const std::string& operation, const std::string& status);
    void record_db_operation_duration(const std::string& operation, double duration_seconds);
    void record_db_connection_error();
    void record_db_query_retry();
    
    // User management metrics recording
    void set_users_total(int count);
    void record_user_operation(const std::string& operation);
    void record_failed_login();
    void set_locked_accounts_total(int count);
    
    // Rate limiting metrics recording
    void record_rate_limit_exceeded(const std::string& endpoint);
    void record_ip_block(const std::string& reason);
    
    // Circuit breaker metrics recording
    void record_circuit_breaker_state_change(const std::string& new_state);
    void set_circuit_breaker_open(bool is_open);
    
    // Utility: RAII timer for automatic duration recording
    class DurationTimer {
    private:
        std::chrono::steady_clock::time_point start_;
        std::function<void(double)> callback_;
        
    public:
        explicit DurationTimer(std::function<void(double)> callback)
            : start_(std::chrono::steady_clock::now()), callback_(std::move(callback)) {}
        
        ~DurationTimer() {
            auto end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration<double>(end - start_).count();
            if (callback_) {
                callback_(duration);
            }
        }
    };
    
    // Create duration timer for authentication
    DurationTimer create_auth_timer(const std::string& method);
    
    // Create duration timer for permission checks
    DurationTimer create_permission_check_timer();
    
    // Create duration timer for database operations
    DurationTimer create_db_operation_timer(const std::string& operation);
};

// Global singleton instance
class PrometheusMetricsManager {
private:
    static std::unique_ptr<PrometheusMetrics> instance_;
    static std::once_flag once_flag_;

public:
    static PrometheusMetrics* get_instance();
    
private:
    PrometheusMetricsManager() = default;
    ~PrometheusMetricsManager() = default;
    PrometheusMetricsManager(const PrometheusMetricsManager&) = delete;
    PrometheusMetricsManager& operator=(const PrometheusMetricsManager&) = delete;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_PROMETHEUS_METRICS_H
