#ifndef JADEVECTORDB_CONNECTION_POOL_H
#define JADEVECTORDB_CONNECTION_POOL_H

#include "lib/error_handling.h"
#include "lib/logging.h"
#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <unordered_map>

#ifdef BUILD_WITH_GRPC
#include <grpcpp/grpcpp.h>
#include "distributed.grpc.pb.h"
#endif

namespace jadevectordb {

/**
 * @brief Configuration for connection pool
 */
struct ConnectionPoolConfig {
    size_t min_connections{2};          // Minimum connections to maintain
    size_t max_connections{20};         // Maximum connections allowed
    size_t initial_connections{5};      // Initial connections to create
    std::chrono::seconds idle_timeout{300};  // Connection idle timeout
    std::chrono::seconds connection_timeout{5};  // Timeout for creating connection
    std::chrono::seconds wait_timeout{10};  // Timeout for waiting for available connection
    bool enable_health_check{true};     // Enable periodic health checks
    std::chrono::seconds health_check_interval{30};  // Health check interval
    bool enable_compression{true};      // Enable gRPC compression
    size_t max_message_size{100 * 1024 * 1024};  // 100MB max message size

    ConnectionPoolConfig() = default;
};

/**
 * @brief Represents a single pooled gRPC connection
 */
struct PooledConnection {
    std::string connection_id;
    std::string target_address;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_used;
    bool in_use;
    bool is_healthy;
    int64_t total_requests;
    int64_t failed_requests;

#ifdef BUILD_WITH_GRPC
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<distributed::DistributedService::Stub> stub;
#endif

    PooledConnection() :
        in_use(false),
        is_healthy(true),
        total_requests(0),
        failed_requests(0) {
        created_at = std::chrono::steady_clock::now();
        last_used = created_at;
    }
};

/**
 * @brief Connection pool for managing gRPC connections to worker nodes
 *
 * Features:
 * - Maintains pool of reusable connections
 * - Automatic connection creation and cleanup
 * - Connection health checking
 * - Load balancing across connections
 * - Thread-safe connection acquisition/release
 * - Idle connection timeout
 * - Maximum connection limit enforcement
 */
class ConnectionPool {
private:
    std::shared_ptr<logging::Logger> logger_;
    ConnectionPoolConfig config_;

    // Pool state
    mutable std::mutex pool_mutex_;
    std::condition_variable pool_cv_;

    // Per-target connection pools
    struct TargetPool {
        std::string target_address;
        std::queue<std::shared_ptr<PooledConnection>> available_connections;
        std::vector<std::shared_ptr<PooledConnection>> all_connections;
        size_t active_connections{0};
        bool shutting_down{false};

        TargetPool() = default;
        explicit TargetPool(const std::string& addr) : target_address(addr) {}
    };

    std::unordered_map<std::string, std::unique_ptr<TargetPool>> target_pools_;

    // Statistics
    struct PoolStatistics {
        int64_t total_acquisitions{0};
        int64_t total_releases{0};
        int64_t total_connections_created{0};
        int64_t total_connections_destroyed{0};
        int64_t failed_acquisitions{0};
        int64_t timeout_acquisitions{0};
        std::chrono::steady_clock::time_point start_time;

        PoolStatistics() {
            start_time = std::chrono::steady_clock::now();
        }
    };

    PoolStatistics stats_;

    // Health checking
    bool health_check_enabled_;
    std::thread health_check_thread_;
    std::atomic<bool> should_stop_health_check_{false};

    bool initialized_;
    bool shutdown_;

public:
    explicit ConnectionPool(const ConnectionPoolConfig& config = ConnectionPoolConfig());
    ~ConnectionPool();

    // ===== Lifecycle =====

    Result<bool> initialize();
    Result<bool> shutdown_pool();
    bool is_initialized() const { return initialized_; }

    // ===== Connection Management =====

    /**
     * @brief Acquire a connection from the pool for a target
     * @param target Target address (host:port)
     * @return Pooled connection or error
     */
    Result<std::shared_ptr<PooledConnection>> acquire(const std::string& target);

    /**
     * @brief Acquire a connection with custom timeout
     * @param target Target address
     * @param timeout Timeout for waiting
     * @return Pooled connection or error
     */
    Result<std::shared_ptr<PooledConnection>> acquire_with_timeout(
        const std::string& target,
        std::chrono::milliseconds timeout
    );

    /**
     * @brief Release a connection back to the pool
     * @param connection Connection to release
     * @param mark_unhealthy Mark connection as unhealthy
     */
    Result<bool> release(std::shared_ptr<PooledConnection> connection, bool mark_unhealthy = false);

    /**
     * @brief Create initial connections for a target
     * @param target Target address
     * @param count Number of connections to create
     */
    Result<bool> warm_up(const std::string& target, size_t count);

    /**
     * @brief Remove all connections for a target
     * @param target Target address
     */
    Result<bool> remove_target(const std::string& target);

    /**
     * @brief Get number of available connections for a target
     * @param target Target address
     */
    size_t get_available_count(const std::string& target) const;

    /**
     * @brief Get total number of connections for a target
     * @param target Target address
     */
    size_t get_total_count(const std::string& target) const;

    /**
     * @brief Get number of active (in-use) connections for a target
     * @param target Target address
     */
    size_t get_active_count(const std::string& target) const;

    // ===== Statistics & Monitoring =====

    struct PoolStats {
        size_t total_targets;
        size_t total_connections;
        size_t available_connections;
        size_t active_connections;
        int64_t total_acquisitions;
        int64_t total_releases;
        int64_t connections_created;
        int64_t connections_destroyed;
        int64_t failed_acquisitions;
        int64_t timeout_acquisitions;
        double acquisition_success_rate;
        int64_t uptime_seconds;
    };

    PoolStats get_statistics() const;

    /**
     * @brief Get statistics for a specific target
     */
    struct TargetStats {
        std::string target_address;
        size_t total_connections;
        size_t available_connections;
        size_t active_connections;
        int64_t total_requests;
        int64_t failed_requests;
        double failure_rate;
    };

    Result<TargetStats> get_target_statistics(const std::string& target) const;

    /**
     * @brief Get health status of all targets
     */
    std::unordered_map<std::string, bool> get_health_status() const;

    // ===== Configuration =====

    ConnectionPoolConfig get_config() const { return config_; }
    Result<bool> update_config(const ConnectionPoolConfig& new_config);

private:
    // Helper methods

    // Get or create target pool
    TargetPool* get_or_create_target_pool(const std::string& target);

    // Create a new connection
    Result<std::shared_ptr<PooledConnection>> create_connection(const std::string& target);

    // Destroy a connection
    void destroy_connection(std::shared_ptr<PooledConnection> connection);

    // Check if connection is healthy
    bool is_connection_healthy(std::shared_ptr<PooledConnection> connection) const;

    // Check if connection is idle
    bool is_connection_idle(std::shared_ptr<PooledConnection> connection) const;

    // Cleanup idle connections
    void cleanup_idle_connections();

    // Health check thread function
    void health_check_loop();

    // Perform health check on a connection
    bool perform_health_check(std::shared_ptr<PooledConnection> connection);

#ifdef BUILD_WITH_GRPC
    // Create gRPC channel
    std::shared_ptr<grpc::Channel> create_grpc_channel(const std::string& target);

    // Create gRPC stub
    std::unique_ptr<distributed::DistributedService::Stub> create_grpc_stub(
        std::shared_ptr<grpc::Channel> channel
    );
#endif

    // Generate unique connection ID
    std::string generate_connection_id(const std::string& target);

    // Statistics helpers
    void record_acquisition(bool success, bool timeout);
    void record_release();
    void record_connection_created();
    void record_connection_destroyed();
};

/**
 * @brief RAII wrapper for automatic connection release
 *
 * Usage:
 *   auto conn = pool.acquire(target);
 *   ScopedConnection scoped(pool, conn.value());
 *   // Use connection
 *   // Connection automatically released when scoped goes out of scope
 */
class ScopedConnection {
private:
    ConnectionPool& pool_;
    std::shared_ptr<PooledConnection> connection_;
    bool released_;
    bool mark_unhealthy_;

public:
    ScopedConnection(ConnectionPool& pool, std::shared_ptr<PooledConnection> connection)
        : pool_(pool), connection_(connection), released_(false), mark_unhealthy_(false) {}

    ~ScopedConnection() {
        if (!released_ && connection_) {
            pool_.release(connection_, mark_unhealthy_);
        }
    }

    // No copy
    ScopedConnection(const ScopedConnection&) = delete;
    ScopedConnection& operator=(const ScopedConnection&) = delete;

    // Allow move
    ScopedConnection(ScopedConnection&& other) noexcept
        : pool_(other.pool_),
          connection_(std::move(other.connection_)),
          released_(other.released_),
          mark_unhealthy_(other.mark_unhealthy_) {
        other.released_ = true;
    }

    std::shared_ptr<PooledConnection> get() { return connection_; }
    std::shared_ptr<PooledConnection> operator->() { return connection_; }

    void mark_as_unhealthy() { mark_unhealthy_ = true; }

    void release() {
        if (!released_ && connection_) {
            pool_.release(connection_, mark_unhealthy_);
            released_ = true;
        }
    }
};

} // namespace jadevectordb

#endif // JADEVECTORDB_CONNECTION_POOL_H
