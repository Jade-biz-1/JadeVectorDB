#include "distributed_write_coordinator.h"
#include <algorithm>
#include <sstream>

namespace jadevectordb {

DistributedWriteCoordinator::DistributedWriteCoordinator(
    std::shared_ptr<ShardingService> sharding_service,
    std::shared_ptr<ReplicationService> replication_service,
    std::shared_ptr<DistributedMasterClient> master_client
) : sharding_service_(sharding_service),
    replication_service_(replication_service),
    master_client_(master_client),
    total_writes_(0),
    successful_writes_(0),
    failed_writes_(0),
    total_replications_(0),
    successful_replications_(0),
    failed_replications_(0),
    async_queue_size_(0),
    total_write_latency_ms_(0),
    total_replication_latency_ms_(0),
    initialized_(false),
    shutdown_(false) {
    logger_ = logging::get_logger("DistributedWriteCoordinator");
}

DistributedWriteCoordinator::~DistributedWriteCoordinator() {
    if (initialized_ && !shutdown_) {
        shutdown_coordinator();
    }
}

// ============================================================================
// Lifecycle Methods
// ============================================================================

Result<bool> DistributedWriteCoordinator::initialize() {
    if (initialized_) {
        return create_error(ErrorCode::INVALID_STATE, "Write coordinator already initialized");
    }

    logger_->info("Initializing distributed write coordinator");
    logger_->info("  Replication factor: " + std::to_string(config_.replication_factor));
    logger_->info("  Async replication: " + std::string(config_.enable_async_replication ? "enabled" : "disabled"));
    logger_->info("  Async workers: " + std::to_string(config_.async_worker_threads));

    if (!sharding_service_) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "ShardingService not provided");
    }

    if (!replication_service_) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "ReplicationService not provided");
    }

    if (!master_client_) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "Master client not provided");
    }

    // Start async replication workers if enabled
    if (config_.enable_async_replication) {
        shutdown_async_ = false;

        for (int i = 0; i < config_.async_worker_threads; ++i) {
            async_workers_.emplace_back(&DistributedWriteCoordinator::async_replication_worker, this);
        }

        logger_->info("Started " + std::to_string(config_.async_worker_threads) + " async replication workers");
    }

    initialized_ = true;
    logger_->info("Distributed write coordinator initialized successfully");
    return true;
}

Result<bool> DistributedWriteCoordinator::shutdown_coordinator() {
    if (shutdown_) {
        return true;
    }

    logger_->info("Shutting down distributed write coordinator");

    // Stop async workers
    if (config_.enable_async_replication) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            shutdown_async_ = true;
        }

        queue_cv_.notify_all();

        for (auto& worker : async_workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        async_workers_.clear();

        logger_->info("Stopped async replication workers");
    }

    shutdown_ = true;
    logger_->info("Distributed write coordinator shut down");
    return true;
}

// ============================================================================
// Write Operations
// ============================================================================

Result<WriteResult> DistributedWriteCoordinator::write_vector(const WriteRequest& request) {
    if (!initialized_ || shutdown_) {
        return create_error(ErrorCode::INVALID_STATE, "Write coordinator not available");
    }

    auto start = std::chrono::steady_clock::now();

    logger_->debug("Writing vector: db=" + request.database_id +
                   ", vector=" + request.vector.id +
                   ", consistency=" + std::to_string(static_cast<int>(request.consistency_level)));

    WriteResult result;
    result.request_id = request.request_id.empty() ? generate_request_id() : request.request_id;

    // Determine target shard
    auto shard_result = determine_target_shard(request.database_id, request.vector.id);
    if (!shard_result.has_value()) {
        record_write(false, 0);
        result.success = false;
        result.error_message = shard_result.error().message;
        return result;
    }

    std::string shard_id = shard_result.value();

    // Get primary worker for shard
    auto worker_result = get_shard_worker(shard_id);
    if (!worker_result.has_value()) {
        record_write(false, 0);
        result.success = false;
        result.error_message = worker_result.error().message;
        return result;
    }

    std::string primary_worker = worker_result.value();

    // Write to primary
    DistributedMasterClient::WriteRequest write_req;
    write_req.shard_id = shard_id;
    write_req.request_id = result.request_id;
    write_req.vector = request.vector;
    write_req.consistency_level = static_cast<ConsistencyLevel>(request.consistency_level);
    write_req.wait_for_replication = request.wait_for_replication;

    auto write_response = master_client_->write_to_shard(primary_worker, write_req);

    auto write_end = std::chrono::steady_clock::now();
    result.write_latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        write_end - start).count();

    if (!write_response.has_value()) {
        record_write(false, result.write_latency_ms);
        result.success = false;
        result.error_message = write_response.error().message;
        return result;
    }

    if (!write_response.value().success) {
        record_write(false, result.write_latency_ms);
        result.success = false;
        result.error_message = write_response.value().error_message;
        return result;
    }

    result.vectors_written = 1;
    result.replicas_acknowledged = 1;  // Primary acknowledged

    // Handle replication based on consistency level
    if (request.consistency_level != WriteConsistencyLevel::EVENTUAL) {
        // Get replica workers
        auto replica_result = get_replica_workers(shard_id, primary_worker);

        if (replica_result.has_value() && !replica_result.value().empty()) {
            auto repl_result = replicate_sync(
                shard_id,
                primary_worker,
                replica_result.value(),
                {request.vector},
                request.consistency_level
            );

            if (repl_result.has_value()) {
                result.replicas_acknowledged += repl_result.value();
            }
        }

        auto repl_end = std::chrono::steady_clock::now();
        result.replication_latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            repl_end - write_end).count();

        // Check if consistency requirements are met
        int total_replicas = 1 + (replica_result.has_value() ? replica_result.value().size() : 0);

        if (!meets_consistency_requirements(result.replicas_acknowledged,
                                           request.consistency_level,
                                           total_replicas)) {
            record_write(false, result.write_latency_ms);
            result.success = false;
            result.error_message = "Failed to meet consistency requirements";
            return result;
        }
    } else {
        // Eventual consistency - async replication
        auto replica_result = get_replica_workers(shard_id, primary_worker);

        if (replica_result.has_value() && !replica_result.value().empty()) {
            replicate_async(shard_id, primary_worker, replica_result.value(), {request.vector});
        }
    }

    result.success = true;
    record_write(true, result.write_latency_ms);

    logger_->debug("Write completed: vector=" + request.vector.id +
                   ", replicas=" + std::to_string(result.replicas_acknowledged) +
                   ", latency=" + std::to_string(result.write_latency_ms) + "ms");

    return result;
}

Result<WriteResult> DistributedWriteCoordinator::batch_write_vectors(const BatchWriteRequest& request) {
    if (!initialized_ || shutdown_) {
        return create_error(ErrorCode::INVALID_STATE, "Write coordinator not available");
    }

    auto start = std::chrono::steady_clock::now();

    logger_->info("Batch writing " + std::to_string(request.vectors.size()) + " vectors");

    WriteResult result;
    result.request_id = request.request_id.empty() ? generate_request_id() : request.request_id;

    // Group vectors by shard
    std::unordered_map<std::string, std::vector<Vector>> vectors_by_shard;

    for (const auto& vector : request.vectors) {
        auto shard_result = determine_target_shard(request.database_id, vector.id);
        if (shard_result.has_value()) {
            vectors_by_shard[shard_result.value()].push_back(vector);
        } else {
            logger_->warn("Failed to determine shard for vector: " + vector.id);
        }
    }

    // Write to each shard
    int total_written = 0;
    int total_acks = 0;

    for (const auto& [shard_id, vectors] : vectors_by_shard) {
        auto worker_result = get_shard_worker(shard_id);
        if (!worker_result.has_value()) {
            logger_->warn("Failed to get worker for shard: " + shard_id);
            continue;
        }

        std::string primary_worker = worker_result.value();

        // Batch write to primary
        auto write_response = master_client_->batch_write_to_shard(
            primary_worker,
            shard_id,
            result.request_id,
            vectors,
            static_cast<ConsistencyLevel>(request.consistency_level)
        );

        if (write_response.has_value() && write_response.value().success) {
            total_written += vectors.size();
            total_acks++;

            // Handle replication
            if (request.consistency_level != WriteConsistencyLevel::EVENTUAL) {
                auto replica_result = get_replica_workers(shard_id, primary_worker);
                if (replica_result.has_value() && !replica_result.value().empty()) {
                    auto repl_result = replicate_sync(
                        shard_id,
                        primary_worker,
                        replica_result.value(),
                        vectors,
                        request.consistency_level
                    );

                    if (repl_result.has_value()) {
                        total_acks += repl_result.value();
                    }
                }
            } else {
                auto replica_result = get_replica_workers(shard_id, primary_worker);
                if (replica_result.has_value() && !replica_result.value().empty()) {
                    replicate_async(shard_id, primary_worker, replica_result.value(), vectors);
                }
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    result.write_latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();

    result.vectors_written = total_written;
    result.replicas_acknowledged = total_acks;
    result.success = (total_written > 0);

    if (!result.success) {
        result.error_message = "Failed to write any vectors";
    }

    record_write(result.success, result.write_latency_ms);

    logger_->info("Batch write completed: written=" + std::to_string(total_written) +
                  "/" + std::to_string(request.vectors.size()) +
                  ", latency=" + std::to_string(result.write_latency_ms) + "ms");

    return result;
}

Result<bool> DistributedWriteCoordinator::delete_vector(
    const std::string& database_id,
    const std::string& vector_id,
    WriteConsistencyLevel consistency_level
) {
    logger_->debug("Deleting vector: " + vector_id);

    auto shard_result = determine_target_shard(database_id, vector_id);
    if (!shard_result.has_value()) {
        return tl::unexpected(shard_result.error());
    }

    std::string shard_id = shard_result.value();

    auto worker_result = get_shard_worker(shard_id);
    if (!worker_result.has_value()) {
        return tl::unexpected(worker_result.error());
    }

    std::string primary_worker = worker_result.value();

    auto result = master_client_->delete_from_shard(
        primary_worker,
        shard_id,
        generate_request_id(),
        {vector_id},
        static_cast<ConsistencyLevel>(consistency_level)
    );

    return result;
}

Result<int> DistributedWriteCoordinator::batch_delete_vectors(
    const std::string& database_id,
    const std::vector<std::string>& vector_ids,
    WriteConsistencyLevel consistency_level
) {
    logger_->info("Batch deleting " + std::to_string(vector_ids.size()) + " vectors");

    // Group by shard
    std::unordered_map<std::string, std::vector<std::string>> ids_by_shard;

    for (const auto& vector_id : vector_ids) {
        auto shard_result = determine_target_shard(database_id, vector_id);
        if (shard_result.has_value()) {
            ids_by_shard[shard_result.value()].push_back(vector_id);
        }
    }

    int total_deleted = 0;

    for (const auto& [shard_id, ids] : ids_by_shard) {
        auto worker_result = get_shard_worker(shard_id);
        if (!worker_result.has_value()) {
            continue;
        }

        auto result = master_client_->delete_from_shard(
            worker_result.value(),
            shard_id,
            generate_request_id(),
            ids,
            static_cast<ConsistencyLevel>(consistency_level)
        );

        if (result.has_value() && result.value()) {
            total_deleted += ids.size();
        }
    }

    return total_deleted;
}

// ============================================================================
// Statistics
// ============================================================================

DistributedWriteCoordinator::WriteStats DistributedWriteCoordinator::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    WriteStats stats;
    stats.total_writes = total_writes_;
    stats.successful_writes = successful_writes_;
    stats.failed_writes = failed_writes_;

    if (total_writes_ > 0) {
        stats.success_rate = static_cast<double>(successful_writes_) / total_writes_;
        stats.avg_write_latency_ms = static_cast<double>(total_write_latency_ms_) / total_writes_;
    } else {
        stats.success_rate = 0.0;
        stats.avg_write_latency_ms = 0.0;
    }

    stats.total_replications = total_replications_;
    stats.successful_replications = successful_replications_;
    stats.failed_replications = failed_replications_;

    if (total_replications_ > 0) {
        stats.replication_success_rate = static_cast<double>(successful_replications_) / total_replications_;
        stats.avg_replication_latency_ms = static_cast<double>(total_replication_latency_ms_) / total_replications_;
    } else {
        stats.replication_success_rate = 0.0;
        stats.avg_replication_latency_ms = 0.0;
    }

    stats.async_queue_size = async_queue_size_;

    return stats;
}

void DistributedWriteCoordinator::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    total_writes_ = 0;
    successful_writes_ = 0;
    failed_writes_ = 0;
    total_replications_ = 0;
    successful_replications_ = 0;
    failed_replications_ = 0;
    total_write_latency_ms_ = 0;
    total_replication_latency_ms_ = 0;
}

Result<bool> DistributedWriteCoordinator::update_config(const CoordinatorConfig& new_config) {
    if (new_config.replication_factor < 1) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "Replication factor must be at least 1");
    }

    if (new_config.async_worker_threads < 1) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "Async worker threads must be at least 1");
    }

    // Note: Changing async workers requires reinitialization
    config_ = new_config;

    logger_->info("Write coordinator configuration updated");
    return true;
}

// ============================================================================
// Private Helper Methods - Write Routing
// ============================================================================

Result<std::string> DistributedWriteCoordinator::determine_target_shard(
    const std::string& database_id,
    const std::string& vector_id
) {
    // Use sharding service to determine target shard
    auto result = sharding_service_->get_shard_for_key(database_id + ":" + vector_id);

    if (!result.has_value()) {
        logger_->warn("Failed to determine shard for vector: " + vector_id);
        return tl::unexpected(result.error());
    }

    return result.value();
}

Result<std::string> DistributedWriteCoordinator::get_shard_worker(const std::string& shard_id) {
    // Get primary worker for shard from sharding service
    auto result = sharding_service_->get_node_for_shard(shard_id);

    if (!result.has_value()) {
        logger_->warn("Failed to get worker for shard: " + shard_id);
        return tl::unexpected(result.error());
    }

    return result.value();
}

Result<std::vector<std::string>> DistributedWriteCoordinator::get_replica_workers(
    const std::string& shard_id,
    const std::string& primary_worker_id
) {
    // Get replica nodes from replication service
    auto result = replication_service_->get_replica_nodes(shard_id);

    if (!result.has_value()) {
        logger_->debug("No replicas found for shard: " + shard_id);
        return std::vector<std::string>();
    }

    // Filter out primary
    std::vector<std::string> replicas;
    for (const auto& node : result.value()) {
        if (node != primary_worker_id) {
            replicas.push_back(node);
        }
    }

    return replicas;
}

// ============================================================================
// Replication Methods
// ============================================================================

Result<int> DistributedWriteCoordinator::replicate_sync(
    const std::string& shard_id,
    const std::string& primary_worker_id,
    const std::vector<std::string>& replica_workers,
    const std::vector<Vector>& vectors,
    WriteConsistencyLevel consistency_level
) {
    auto start = std::chrono::steady_clock::now();

    int required_acks = calculate_required_acks(consistency_level, replica_workers.size() + 1);
    int acks_received = 0;

    logger_->debug("Synchronous replication: shard=" + shard_id +
                   ", replicas=" + std::to_string(replica_workers.size()) +
                   ", required_acks=" + std::to_string(required_acks));

    // Replicate to all replicas (could be done in parallel for better performance)
    for (const auto& worker_id : replica_workers) {
        auto result = replicate_to_worker(worker_id, shard_id, vectors);

        if (result.has_value() && result.value()) {
            acks_received++;
            record_replication(true, 0);
        } else {
            record_replication(false, 0);
            logger_->warn("Replication failed to worker: " + worker_id);
        }

        // Early exit if we have enough acks
        if (acks_received >= required_acks - 1) {  // -1 because primary is already counted
            break;
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    logger_->debug("Replication completed: acks=" + std::to_string(acks_received) +
                   "/" + std::to_string(required_acks - 1) +
                   ", latency=" + std::to_string(latency_ms) + "ms");

    return acks_received;
}

Result<bool> DistributedWriteCoordinator::replicate_async(
    const std::string& shard_id,
    const std::string& primary_worker_id,
    const std::vector<std::string>& replica_workers,
    const std::vector<Vector>& vectors
) {
    if (!config_.enable_async_replication) {
        return true;  // Async disabled, skip
    }

    ReplicationTask task;
    task.task_id = generate_request_id();
    task.shard_id = shard_id;
    task.source_worker_id = primary_worker_id;
    task.target_worker_ids = replica_workers;
    task.vectors = vectors;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        if (replication_queue_.size() >= config_.async_queue_max_size) {
            logger_->warn("Async replication queue full, dropping task");
            return create_error(ErrorCode::RESOURCE_EXHAUSTED, "Replication queue full");
        }

        replication_queue_.push(task);
        update_async_queue_size();
    }

    queue_cv_.notify_one();

    logger_->debug("Enqueued async replication task: " + task.task_id);

    return true;
}

void DistributedWriteCoordinator::async_replication_worker() {
    logger_->debug("Async replication worker started");

    while (!shutdown_async_) {
        ReplicationTask task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            queue_cv_.wait_for(lock, config_.async_flush_interval, [this] {
                return !replication_queue_.empty() || shutdown_async_;
            });

            if (shutdown_async_ && replication_queue_.empty()) {
                break;
            }

            if (!replication_queue_.empty()) {
                task = replication_queue_.front();
                replication_queue_.pop();
                update_async_queue_size();
            } else {
                continue;
            }
        }

        // Process task
        bool success = process_replication_task(task);

        if (!success && task.retry_count < config_.max_retry_attempts) {
            // Re-enqueue for retry
            task.retry_count++;

            std::lock_guard<std::mutex> lock(queue_mutex_);
            replication_queue_.push(task);
            update_async_queue_size();

            logger_->debug("Replication task retry " + std::to_string(task.retry_count) +
                          "/" + std::to_string(config_.max_retry_attempts));
        }
    }

    logger_->debug("Async replication worker stopped");
}

bool DistributedWriteCoordinator::process_replication_task(const ReplicationTask& task) {
    logger_->debug("Processing replication task: " + task.task_id);

    bool all_success = true;

    for (const auto& worker_id : task.target_worker_ids) {
        auto result = replicate_to_worker(worker_id, task.shard_id, task.vectors);

        if (result.has_value() && result.value()) {
            record_replication(true, 0);
        } else {
            record_replication(false, 0);
            all_success = false;
        }
    }

    return all_success;
}

Result<bool> DistributedWriteCoordinator::replicate_to_worker(
    const std::string& worker_id,
    const std::string& shard_id,
    const std::vector<Vector>& vectors
) {
    DistributedMasterClient::ReplicationRequest req;
    req.shard_id = shard_id;
    req.source_node_id = "master";  // TODO: Get actual master node ID
    req.replication_type = ReplicationType::FULL;
    req.vectors = vectors;
    req.from_version = 0;
    req.to_version = 0;

    auto result = master_client_->replicate_data(worker_id, req);

    return result.has_value() && result.value().success;
}

// ============================================================================
// Conflict Resolution
// ============================================================================

Result<Vector> DistributedWriteCoordinator::resolve_conflict(
    const Vector& existing_vector,
    const Vector& new_vector
) {
    if (!config_.enable_conflict_resolution) {
        return new_vector;  // Always use new vector
    }

    if (config_.conflict_resolution_strategy == "last_write_wins") {
        return resolve_lww(existing_vector, new_vector);
    }

    // Default: use new vector
    return new_vector;
}

Vector DistributedWriteCoordinator::resolve_lww(
    const Vector& existing_vector,
    const Vector& new_vector
) {
    // Compare timestamps (assuming vectors have timestamps in metadata)
    // For simplicity, just return new vector
    return new_vector;
}

// ============================================================================
// Consistency Level Helpers
// ============================================================================

int DistributedWriteCoordinator::calculate_required_acks(
    WriteConsistencyLevel level,
    int total_replicas
) const {
    switch (level) {
        case WriteConsistencyLevel::STRONG:
            return total_replicas;  // All replicas

        case WriteConsistencyLevel::QUORUM:
            return (total_replicas / 2) + 1;  // Majority

        case WriteConsistencyLevel::EVENTUAL:
            return 1;  // Just primary

        default:
            return 1;
    }
}

bool DistributedWriteCoordinator::meets_consistency_requirements(
    int acks_received,
    WriteConsistencyLevel level,
    int total_replicas
) const {
    int required = calculate_required_acks(level, total_replicas);
    return acks_received >= required;
}

// ============================================================================
// Statistics Helpers
// ============================================================================

void DistributedWriteCoordinator::record_write(bool success, int64_t latency_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    total_writes_++;
    if (success) {
        successful_writes_++;
    } else {
        failed_writes_++;
    }
    total_write_latency_ms_ += latency_ms;
}

void DistributedWriteCoordinator::record_replication(bool success, int64_t latency_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    total_replications_++;
    if (success) {
        successful_replications_++;
    } else {
        failed_replications_++;
    }
    total_replication_latency_ms_ += latency_ms;
}

void DistributedWriteCoordinator::update_async_queue_size() {
    async_queue_size_ = replication_queue_.size();
}

// ============================================================================
// Utility
// ============================================================================

std::string DistributedWriteCoordinator::generate_request_id() const {
    static std::atomic<int64_t> counter{0};

    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();

    std::stringstream ss;
    ss << "w_" << timestamp << "_" << counter++;
    return ss.str();
}

} // namespace jadevectordb
