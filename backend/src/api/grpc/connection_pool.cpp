#include "connection_pool.h"
#include <sstream>
#include <iomanip>
#include <random>

namespace jadevectordb {

ConnectionPool::ConnectionPool(const ConnectionPoolConfig& config)
    : config_(config),
      health_check_enabled_(config.enable_health_check),
      initialized_(false),
      shutdown_(false) {
    logger_ = logging::get_logger("ConnectionPool");
}

ConnectionPool::~ConnectionPool() {
    if (initialized_ && !shutdown_) {
        shutdown_pool();
    }
}

// ============================================================================
// Lifecycle Methods
// ============================================================================

Result<bool> ConnectionPool::initialize() {
    if (initialized_) {
        return create_error(ErrorCode::INVALID_STATE, "Connection pool already initialized");
    }

    logger_->info("Initializing connection pool");
    logger_->info("  Min connections: " + std::to_string(config_.min_connections));
    logger_->info("  Max connections: " + std::to_string(config_.max_connections));
    logger_->info("  Initial connections: " + std::to_string(config_.initial_connections));

    // Validate configuration
    if (config_.min_connections > config_.max_connections) {
        return create_error(ErrorCode::INVALID_ARGUMENT,
                          "Min connections cannot exceed max connections");
    }

    if (config_.initial_connections > config_.max_connections) {
        return create_error(ErrorCode::INVALID_ARGUMENT,
                          "Initial connections cannot exceed max connections");
    }

    // Start health check thread if enabled
    if (health_check_enabled_) {
        should_stop_health_check_ = false;
        health_check_thread_ = std::thread(&ConnectionPool::health_check_loop, this);
        logger_->info("Health check thread started");
    }

    initialized_ = true;
    logger_->info("Connection pool initialized successfully");
    return true;
}

Result<bool> ConnectionPool::shutdown_pool() {
    if (shutdown_) {
        return true;
    }

    logger_->info("Shutting down connection pool");

    // Stop health check thread
    if (health_check_enabled_ && health_check_thread_.joinable()) {
        should_stop_health_check_ = true;
        pool_cv_.notify_all();
        health_check_thread_.join();
        logger_->info("Health check thread stopped");
    }

    // Close all connections
    std::lock_guard<std::mutex> lock(pool_mutex_);

    for (auto& [target, pool] : target_pools_) {
        pool->shutting_down = true;

        // Destroy all connections
        for (auto& conn : pool->all_connections) {
            if (conn) {
                destroy_connection(conn);
            }
        }

        pool->available_connections = std::queue<std::shared_ptr<PooledConnection>>();
        pool->all_connections.clear();

        logger_->info("Closed all connections for target: " + target);
    }

    target_pools_.clear();

    shutdown_ = true;
    logger_->info("Connection pool shut down");
    return true;
}

// ============================================================================
// Connection Management
// ============================================================================

Result<std::shared_ptr<PooledConnection>> ConnectionPool::acquire(const std::string& target) {
    return acquire_with_timeout(target, config_.wait_timeout);
}

Result<std::shared_ptr<PooledConnection>> ConnectionPool::acquire_with_timeout(
    const std::string& target,
    std::chrono::milliseconds timeout
) {
    if (!initialized_ || shutdown_) {
        return create_error(ErrorCode::INVALID_STATE, "Connection pool not available");
    }

    logger_->debug("Acquiring connection for target: " + target);

    auto deadline = std::chrono::steady_clock::now() + timeout;

    std::unique_lock<std::mutex> lock(pool_mutex_);

    auto* pool = get_or_create_target_pool(target);
    if (!pool) {
        record_acquisition(false, false);
        return create_error(ErrorCode::INTERNAL_ERROR, "Failed to create target pool");
    }

    // Try to get an available connection or create a new one
    while (true) {
        // Check if we have available connections
        if (!pool->available_connections.empty()) {
            auto conn = pool->available_connections.front();
            pool->available_connections.pop();

            // Check if connection is still healthy
            if (is_connection_healthy(conn)) {
                conn->in_use = true;
                conn->last_used = std::chrono::steady_clock::now();
                pool->active_connections++;

                record_acquisition(true, false);
                logger_->debug("Acquired existing connection: " + conn->connection_id);
                return conn;
            } else {
                // Connection unhealthy, destroy it
                logger_->warn("Connection unhealthy, destroying: " + conn->connection_id);
                destroy_connection(conn);

                auto it = std::find(pool->all_connections.begin(),
                                   pool->all_connections.end(), conn);
                if (it != pool->all_connections.end()) {
                    pool->all_connections.erase(it);
                }
                continue;
            }
        }

        // No available connections, try to create a new one
        if (pool->all_connections.size() < config_.max_connections) {
            lock.unlock();
            auto new_conn = create_connection(target);
            lock.lock();

            if (new_conn.has_value()) {
                auto conn = new_conn.value();
                conn->in_use = true;
                pool->all_connections.push_back(conn);
                pool->active_connections++;

                record_acquisition(true, false);
                logger_->debug("Created new connection: " + conn->connection_id);
                return conn;
            } else {
                logger_->error("Failed to create connection: " + new_conn.error().message);
                // Fall through to wait
            }
        }

        // Wait for a connection to become available or timeout
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            record_acquisition(false, true);
            return create_error(ErrorCode::TIMEOUT,
                              "Timeout waiting for connection to target: " + target);
        }

        auto wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        if (pool_cv_.wait_for(lock, wait_time) == std::cv_status::timeout) {
            record_acquisition(false, true);
            return create_error(ErrorCode::TIMEOUT,
                              "Timeout waiting for connection to target: " + target);
        }
    }
}

Result<bool> ConnectionPool::release(std::shared_ptr<PooledConnection> connection,
                                    bool mark_unhealthy) {
    if (!connection) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "Null connection");
    }

    logger_->debug("Releasing connection: " + connection->connection_id +
                   (mark_unhealthy ? " (unhealthy)" : ""));

    std::lock_guard<std::mutex> lock(pool_mutex_);

    auto it = target_pools_.find(connection->target_address);
    if (it == target_pools_.end()) {
        return create_error(ErrorCode::NOT_FOUND, "Target pool not found");
    }

    auto* pool = it->second.get();

    connection->in_use = false;

    if (mark_unhealthy) {
        connection->is_healthy = false;
    }

    if (pool->shutting_down || !connection->is_healthy) {
        // Don't return to pool, destroy it
        destroy_connection(connection);

        auto conn_it = std::find(pool->all_connections.begin(),
                                pool->all_connections.end(), connection);
        if (conn_it != pool->all_connections.end()) {
            pool->all_connections.erase(conn_it);
        }
    } else {
        // Return to available pool
        connection->last_used = std::chrono::steady_clock::now();
        pool->available_connections.push(connection);
    }

    if (pool->active_connections > 0) {
        pool->active_connections--;
    }

    record_release();
    pool_cv_.notify_one();

    return true;
}

Result<bool> ConnectionPool::warm_up(const std::string& target, size_t count) {
    logger_->info("Warming up " + std::to_string(count) + " connections for target: " + target);

    for (size_t i = 0; i < count; ++i) {
        auto conn_result = acquire(target);
        if (conn_result.has_value()) {
            release(conn_result.value());
        } else {
            logger_->warn("Failed to create connection " + std::to_string(i+1) + "/" +
                         std::to_string(count));
        }
    }

    logger_->info("Warm-up completed for target: " + target);
    return true;
}

Result<bool> ConnectionPool::remove_target(const std::string& target) {
    logger_->info("Removing target: " + target);

    std::lock_guard<std::mutex> lock(pool_mutex_);

    auto it = target_pools_.find(target);
    if (it == target_pools_.end()) {
        return create_error(ErrorCode::NOT_FOUND, "Target not found");
    }

    auto* pool = it->second.get();
    pool->shutting_down = true;

    // Destroy all connections
    for (auto& conn : pool->all_connections) {
        if (conn && !conn->in_use) {
            destroy_connection(conn);
        }
    }

    target_pools_.erase(it);

    logger_->info("Target removed: " + target);
    return true;
}

// ============================================================================
// Query Methods
// ============================================================================

size_t ConnectionPool::get_available_count(const std::string& target) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    auto it = target_pools_.find(target);
    if (it == target_pools_.end()) {
        return 0;
    }

    return it->second->available_connections.size();
}

size_t ConnectionPool::get_total_count(const std::string& target) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    auto it = target_pools_.find(target);
    if (it == target_pools_.end()) {
        return 0;
    }

    return it->second->all_connections.size();
}

size_t ConnectionPool::get_active_count(const std::string& target) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    auto it = target_pools_.find(target);
    if (it == target_pools_.end()) {
        return 0;
    }

    return it->second->active_connections;
}

// ============================================================================
// Statistics
// ============================================================================

ConnectionPool::PoolStats ConnectionPool::get_statistics() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    PoolStats stats;
    stats.total_targets = target_pools_.size();
    stats.total_connections = 0;
    stats.available_connections = 0;
    stats.active_connections = 0;

    for (const auto& [_, pool] : target_pools_) {
        stats.total_connections += pool->all_connections.size();
        stats.available_connections += pool->available_connections.size();
        stats.active_connections += pool->active_connections;
    }

    stats.total_acquisitions = stats_.total_acquisitions;
    stats.total_releases = stats_.total_releases;
    stats.connections_created = stats_.total_connections_created;
    stats.connections_destroyed = stats_.total_connections_destroyed;
    stats.failed_acquisitions = stats_.failed_acquisitions;
    stats.timeout_acquisitions = stats_.timeout_acquisitions;

    if (stats_.total_acquisitions > 0) {
        stats.acquisition_success_rate =
            1.0 - (static_cast<double>(stats_.failed_acquisitions) / stats_.total_acquisitions);
    } else {
        stats.acquisition_success_rate = 1.0;
    }

    auto now = std::chrono::steady_clock::now();
    stats.uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(
        now - stats_.start_time).count();

    return stats;
}

// ============================================================================
// Helper Methods
// ============================================================================

ConnectionPool::TargetPool* ConnectionPool::get_or_create_target_pool(const std::string& target) {
    auto it = target_pools_.find(target);
    if (it != target_pools_.end()) {
        return it->second.get();
    }

    // Create new target pool
    auto pool = std::make_unique<TargetPool>(target);
    auto* pool_ptr = pool.get();
    target_pools_[target] = std::move(pool);

    logger_->info("Created new target pool for: " + target);
    return pool_ptr;
}

Result<std::shared_ptr<PooledConnection>> ConnectionPool::create_connection(
    const std::string& target
) {
    logger_->debug("Creating new connection for target: " + target);

    auto connection = std::make_shared<PooledConnection>();
    connection->connection_id = generate_connection_id(target);
    connection->target_address = target;
    connection->created_at = std::chrono::steady_clock::now();
    connection->last_used = connection->created_at;

#ifdef BUILD_WITH_GRPC
    connection->channel = create_grpc_channel(target);
    if (!connection->channel) {
        return create_error(ErrorCode::CONNECTION_ERROR,
                          "Failed to create gRPC channel for: " + target);
    }

    connection->stub = create_grpc_stub(connection->channel);
    if (!connection->stub) {
        return create_error(ErrorCode::CONNECTION_ERROR,
                          "Failed to create gRPC stub for: " + target);
    }
#endif

    record_connection_created();
    logger_->debug("Connection created: " + connection->connection_id);
    return connection;
}

void ConnectionPool::destroy_connection(std::shared_ptr<PooledConnection> connection) {
    if (!connection) {
        return;
    }

    logger_->debug("Destroying connection: " + connection->connection_id);

#ifdef BUILD_WITH_GRPC
    connection->stub.reset();
    connection->channel.reset();
#endif

    record_connection_destroyed();
}

bool ConnectionPool::is_connection_healthy(std::shared_ptr<PooledConnection> connection) const {
    if (!connection || !connection->is_healthy) {
        return false;
    }

#ifdef BUILD_WITH_GRPC
    if (!connection->channel) {
        return false;
    }

    // Check channel state
    auto state = connection->channel->GetState(false);
    if (state == GRPC_CHANNEL_SHUTDOWN || state == GRPC_CHANNEL_TRANSIENT_FAILURE) {
        return false;
    }
#endif

    return true;
}

bool ConnectionPool::is_connection_idle(std::shared_ptr<PooledConnection> connection) const {
    if (!connection || connection->in_use) {
        return false;
    }

    auto now = std::chrono::steady_clock::now();
    auto idle_time = std::chrono::duration_cast<std::chrono::seconds>(
        now - connection->last_used);

    return idle_time >= config_.idle_timeout;
}

void ConnectionPool::cleanup_idle_connections() {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    for (auto& [target, pool] : target_pools_) {
        std::vector<std::shared_ptr<PooledConnection>> to_destroy;

        // Check all connections
        for (auto& conn : pool->all_connections) {
            if (!conn->in_use && is_connection_idle(conn)) {
                to_destroy.push_back(conn);
            }
        }

        // Destroy idle connections (but maintain minimum)
        for (auto& conn : to_destroy) {
            if (pool->all_connections.size() > config_.min_connections) {
                // Remove from available queue
                std::queue<std::shared_ptr<PooledConnection>> temp_queue;
                while (!pool->available_connections.empty()) {
                    auto c = pool->available_connections.front();
                    pool->available_connections.pop();

                    if (c != conn) {
                        temp_queue.push(c);
                    }
                }
                pool->available_connections = temp_queue;

                // Remove from all connections
                auto it = std::find(pool->all_connections.begin(),
                                   pool->all_connections.end(), conn);
                if (it != pool->all_connections.end()) {
                    pool->all_connections.erase(it);
                }

                destroy_connection(conn);
                logger_->debug("Destroyed idle connection: " + conn->connection_id);
            }
        }
    }
}

void ConnectionPool::health_check_loop() {
    logger_->info("Health check loop started");

    while (!should_stop_health_check_) {
        std::this_thread::sleep_for(config_.health_check_interval);

        if (should_stop_health_check_) {
            break;
        }

        // Cleanup idle connections
        cleanup_idle_connections();

        // Perform health checks
        // TODO: Implement actual health check RPC calls
    }

    logger_->info("Health check loop stopped");
}

#ifdef BUILD_WITH_GRPC
std::shared_ptr<grpc::Channel> ConnectionPool::create_grpc_channel(const std::string& target) {
    grpc::ChannelArguments args;

    if (config_.max_message_size > 0) {
        args.SetMaxSendMessageSize(config_.max_message_size);
        args.SetMaxReceiveMessageSize(config_.max_message_size);
    }

    if (config_.enable_compression) {
        args.SetCompressionAlgorithm(GRPC_COMPRESS_GZIP);
    }

    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 5000);
    args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);

    return grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), args);
}

std::unique_ptr<distributed::DistributedService::Stub> ConnectionPool::create_grpc_stub(
    std::shared_ptr<grpc::Channel> channel
) {
    return distributed::DistributedService::NewStub(channel);
}
#endif

std::string ConnectionPool::generate_connection_id(const std::string& target) {
    static std::atomic<int64_t> counter{0};

    std::stringstream ss;
    ss << "conn_" << target << "_" << counter++;
    return ss.str();
}

void ConnectionPool::record_acquisition(bool success, bool timeout) {
    stats_.total_acquisitions++;
    if (!success) {
        stats_.failed_acquisitions++;
        if (timeout) {
            stats_.timeout_acquisitions++;
        }
    }
}

void ConnectionPool::record_release() {
    stats_.total_releases++;
}

void ConnectionPool::record_connection_created() {
    stats_.total_connections_created++;
}

void ConnectionPool::record_connection_destroyed() {
    stats_.total_connections_destroyed++;
}

} // namespace jadevectordb
