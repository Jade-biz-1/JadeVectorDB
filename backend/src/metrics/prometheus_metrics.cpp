#include "metrics/prometheus_metrics.h"
#include "lib/logging.h"

namespace jadevectordb {

// Static member initialization
std::unique_ptr<PrometheusMetrics> PrometheusMetricsManager::instance_ = nullptr;
std::once_flag PrometheusMetricsManager::once_flag_;

PrometheusMetrics::PrometheusMetrics() {
    registry_ = MetricsManager::get_registry();
    initialize();
}

void PrometheusMetrics::initialize() {
    auto logger = logging::LoggerManager::get_logger("PrometheusMetrics");
    LOG_INFO(logger, "Initializing Prometheus metrics...");
    
    // Define histogram buckets for duration metrics (in seconds)
    // Buckets: 1ms, 5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s, 10s
    std::vector<double> duration_buckets = {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0};
    
    // Authentication metrics
    auth_requests_total_ = registry_->register_counter(
        "jadevectordb_auth_requests_total",
        "Total number of authentication requests"
    );
    
    auth_duration_seconds_ = registry_->register_histogram(
        "jadevectordb_auth_duration_seconds",
        "Duration of authentication requests in seconds",
        duration_buckets
    );
    
    auth_errors_total_ = registry_->register_counter(
        "jadevectordb_auth_errors_total",
        "Total number of authentication errors"
    );
    
    active_sessions_ = registry_->register_gauge(
        "jadevectordb_active_sessions",
        "Number of currently active sessions"
    );
    
    // Permission check metrics
    permission_checks_total_ = registry_->register_counter(
        "jadevectordb_permission_checks_total",
        "Total number of permission checks"
    );
    
    permission_check_duration_seconds_ = registry_->register_histogram(
        "jadevectordb_permission_check_duration_seconds",
        "Duration of permission checks in seconds",
        duration_buckets
    );
    
    permission_cache_hits_total_ = registry_->register_counter(
        "jadevectordb_permission_cache_hits_total",
        "Total number of permission cache hits"
    );
    
    permission_cache_misses_total_ = registry_->register_counter(
        "jadevectordb_permission_cache_misses_total",
        "Total number of permission cache misses"
    );
    
    // Database operation metrics
    db_operations_total_ = registry_->register_counter(
        "jadevectordb_db_operations_total",
        "Total number of database operations"
    );
    
    db_operation_duration_seconds_ = registry_->register_histogram(
        "jadevectordb_db_operation_duration_seconds",
        "Duration of database operations in seconds",
        duration_buckets
    );
    
    db_connection_errors_total_ = registry_->register_counter(
        "jadevectordb_db_connection_errors_total",
        "Total number of database connection errors"
    );
    
    db_query_retries_total_ = registry_->register_counter(
        "jadevectordb_db_query_retries_total",
        "Total number of database query retries"
    );
    
    // User management metrics
    users_total_ = registry_->register_gauge(
        "jadevectordb_users_total",
        "Total number of users in the system"
    );
    
    user_operations_total_ = registry_->register_counter(
        "jadevectordb_user_operations_total",
        "Total number of user management operations"
    );
    
    failed_logins_total_ = registry_->register_counter(
        "jadevectordb_failed_logins_total",
        "Total number of failed login attempts"
    );
    
    locked_accounts_total_ = registry_->register_gauge(
        "jadevectordb_locked_accounts_total",
        "Number of currently locked accounts"
    );
    
    // Rate limiting metrics
    rate_limit_exceeded_total_ = registry_->register_counter(
        "jadevectordb_rate_limit_exceeded_total",
        "Total number of rate limit violations"
    );
    
    ip_blocks_total_ = registry_->register_counter(
        "jadevectordb_ip_blocks_total",
        "Total number of IP blocks"
    );
    
    // Circuit breaker metrics
    circuit_breaker_state_changes_total_ = registry_->register_counter(
        "jadevectordb_circuit_breaker_state_changes_total",
        "Total number of circuit breaker state changes"
    );
    
    circuit_breaker_open_ = registry_->register_gauge(
        "jadevectordb_circuit_breaker_open",
        "Circuit breaker is open (1) or closed (0)"
    );
    
    LOG_INFO(logger, "Prometheus metrics initialized successfully");
}

std::string PrometheusMetrics::get_metrics() const {
    return registry_->to_prometheus_format();
}

// Authentication metrics recording
void PrometheusMetrics::record_auth_request(const std::string& method, const std::string& status) {
    if (auth_requests_total_) {
        auth_requests_total_->increment();
    }
}

void PrometheusMetrics::record_auth_duration(const std::string& method, double duration_seconds) {
    if (auth_duration_seconds_) {
        auth_duration_seconds_->observe(duration_seconds);
    }
}

void PrometheusMetrics::record_auth_error(const std::string& method, const std::string& error_type) {
    if (auth_errors_total_) {
        auth_errors_total_->increment();
    }
}

void PrometheusMetrics::set_active_sessions(int count) {
    if (active_sessions_) {
        active_sessions_->set(static_cast<double>(count));
    }
}

void PrometheusMetrics::increment_active_sessions() {
    if (active_sessions_) {
        active_sessions_->increment();
    }
}

void PrometheusMetrics::decrement_active_sessions() {
    if (active_sessions_) {
        active_sessions_->decrement();
    }
}

// Permission check metrics recording
void PrometheusMetrics::record_permission_check(const std::string& permission, const std::string& result) {
    if (permission_checks_total_) {
        permission_checks_total_->increment();
    }
}

void PrometheusMetrics::record_permission_check_duration(double duration_seconds) {
    if (permission_check_duration_seconds_) {
        permission_check_duration_seconds_->observe(duration_seconds);
    }
}

void PrometheusMetrics::record_permission_cache_hit() {
    if (permission_cache_hits_total_) {
        permission_cache_hits_total_->increment();
    }
}

void PrometheusMetrics::record_permission_cache_miss() {
    if (permission_cache_misses_total_) {
        permission_cache_misses_total_->increment();
    }
}

// Database operation metrics recording
void PrometheusMetrics::record_db_operation(const std::string& operation, const std::string& status) {
    if (db_operations_total_) {
        db_operations_total_->increment();
    }
}

void PrometheusMetrics::record_db_operation_duration(const std::string& operation, double duration_seconds) {
    if (db_operation_duration_seconds_) {
        db_operation_duration_seconds_->observe(duration_seconds);
    }
}

void PrometheusMetrics::record_db_connection_error() {
    if (db_connection_errors_total_) {
        db_connection_errors_total_->increment();
    }
}

void PrometheusMetrics::record_db_query_retry() {
    if (db_query_retries_total_) {
        db_query_retries_total_->increment();
    }
}

// User management metrics recording
void PrometheusMetrics::set_users_total(int count) {
    if (users_total_) {
        users_total_->set(static_cast<double>(count));
    }
}

void PrometheusMetrics::record_user_operation(const std::string& operation) {
    if (user_operations_total_) {
        user_operations_total_->increment();
    }
}

void PrometheusMetrics::record_failed_login() {
    if (failed_logins_total_) {
        failed_logins_total_->increment();
    }
}

void PrometheusMetrics::set_locked_accounts_total(int count) {
    if (locked_accounts_total_) {
        locked_accounts_total_->set(static_cast<double>(count));
    }
}

// Rate limiting metrics recording
void PrometheusMetrics::record_rate_limit_exceeded(const std::string& endpoint) {
    if (rate_limit_exceeded_total_) {
        rate_limit_exceeded_total_->increment();
    }
}

void PrometheusMetrics::record_ip_block(const std::string& reason) {
    if (ip_blocks_total_) {
        ip_blocks_total_->increment();
    }
}

// Circuit breaker metrics recording
void PrometheusMetrics::record_circuit_breaker_state_change(const std::string& new_state) {
    if (circuit_breaker_state_changes_total_) {
        circuit_breaker_state_changes_total_->increment();
    }
}

void PrometheusMetrics::set_circuit_breaker_open(bool is_open) {
    if (circuit_breaker_open_) {
        circuit_breaker_open_->set(is_open ? 1.0 : 0.0);
    }
}

// Duration timer implementations
PrometheusMetrics::DurationTimer PrometheusMetrics::create_auth_timer(const std::string& method) {
    return DurationTimer([this, method](double duration) {
        record_auth_duration(method, duration);
    });
}

PrometheusMetrics::DurationTimer PrometheusMetrics::create_permission_check_timer() {
    return DurationTimer([this](double duration) {
        record_permission_check_duration(duration);
    });
}

PrometheusMetrics::DurationTimer PrometheusMetrics::create_db_operation_timer(const std::string& operation) {
    return DurationTimer([this, operation](double duration) {
        record_db_operation_duration(operation, duration);
    });
}

// Singleton implementation
PrometheusMetrics* PrometheusMetricsManager::get_instance() {
    std::call_once(once_flag_, []() {
        instance_ = std::make_unique<PrometheusMetrics>();
    });
    return instance_.get();
}

} // namespace jadevectordb
