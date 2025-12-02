#ifndef JADEVECTORDB_DISTRIBUTED_WRITE_COORDINATOR_H
#define JADEVECTORDB_DISTRIBUTED_WRITE_COORDINATOR_H

#include "lib/error_handling.h"
#include "lib/logging.h"
#include "models/vector.h"
#include "models/database.h"
#include "sharding_service.h"
#include "replication_service.h"
#include "api/grpc/distributed_master_client.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>

namespace jadevectordb {

/**
 * @brief Consistency level for write operations
 */
enum class WriteConsistencyLevel {
    STRONG,      // Wait for all replicas to acknowledge
    QUORUM,      // Wait for majority of replicas
    EVENTUAL     // Async replication, acknowledge immediately
};

/**
 * @brief Write operation request
 */
struct WriteRequest {
    std::string request_id;
    std::string database_id;
    Vector vector;
    WriteConsistencyLevel consistency_level{WriteConsistencyLevel::QUORUM};
    bool wait_for_replication{true};
    std::chrono::milliseconds timeout{5000};

    WriteRequest() = default;
};

/**
 * @brief Batch write operation request
 */
struct BatchWriteRequest {
    std::string request_id;
    std::string database_id;
    std::vector<Vector> vectors;
    WriteConsistencyLevel consistency_level{WriteConsistencyLevel::QUORUM};
    bool wait_for_replication{true};
    std::chrono::milliseconds timeout{10000};

    BatchWriteRequest() = default;
};

/**
 * @brief Write operation result
 */
struct WriteResult {
    std::string request_id;
    bool success;
    std::string error_message;
    int vectors_written;
    int replicas_acknowledged;
    int64_t write_latency_ms;
    int64_t replication_latency_ms;

    WriteResult() :
        success(false),
        vectors_written(0),
        replicas_acknowledged(0),
        write_latency_ms(0),
        replication_latency_ms(0) {}
};

/**
 * @brief Pending replication task
 */
struct ReplicationTask {
    std::string task_id;
    std::string shard_id;
    std::string source_worker_id;
    std::vector<std::string> target_worker_ids;
    std::vector<Vector> vectors;
    std::chrono::steady_clock::time_point created_at;
    int retry_count{0};

    ReplicationTask() {
        created_at = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Coordinates distributed write operations
 *
 * Responsibilities:
 * - Route write requests to appropriate shards/workers
 * - Handle synchronous and asynchronous replication
 * - Manage consistency levels (STRONG, QUORUM, EVENTUAL)
 * - Conflict resolution with version vectors
 * - Replication queue for async writes
 * - Write statistics and monitoring
 */
class DistributedWriteCoordinator {
private:
    std::shared_ptr<logging::Logger> logger_;
    std::shared_ptr<ShardingService> sharding_service_;
    std::shared_ptr<ReplicationService> replication_service_;
    std::shared_ptr<DistributedMasterClient> master_client_;

    // Configuration
    struct CoordinatorConfig {
        int replication_factor{3};
        bool enable_async_replication{true};
        size_t async_queue_max_size{10000};
        int async_worker_threads{4};
        std::chrono::milliseconds async_flush_interval{100};
        int max_retry_attempts{3};
        std::chrono::milliseconds retry_backoff_base{100};
        bool enable_conflict_resolution{true};
        std::string conflict_resolution_strategy{"last_write_wins"};

        CoordinatorConfig() = default;
    };

    CoordinatorConfig config_;

    // Async replication queue
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<ReplicationTask> replication_queue_;
    std::vector<std::thread> async_workers_;
    std::atomic<bool> shutdown_async_{false};

    // Statistics
    mutable std::mutex stats_mutex_;
    int64_t total_writes_;
    int64_t successful_writes_;
    int64_t failed_writes_;
    int64_t total_replications_;
    int64_t successful_replications_;
    int64_t failed_replications_;
    int64_t async_queue_size_;
    int64_t total_write_latency_ms_;
    int64_t total_replication_latency_ms_;

    bool initialized_;
    bool shutdown_;

public:
    explicit DistributedWriteCoordinator(
        std::shared_ptr<ShardingService> sharding_service,
        std::shared_ptr<ReplicationService> replication_service,
        std::shared_ptr<DistributedMasterClient> master_client
    );

    ~DistributedWriteCoordinator();

    // ===== Lifecycle =====

    Result<bool> initialize();
    Result<bool> shutdown_coordinator();
    bool is_initialized() const { return initialized_; }

    // ===== Write Operations =====

    /**
     * @brief Write a single vector to the distributed system
     * @param request Write request
     * @return Write result
     */
    Result<WriteResult> write_vector(const WriteRequest& request);

    /**
     * @brief Write multiple vectors in a batch
     * @param request Batch write request
     * @return Write result
     */
    Result<WriteResult> batch_write_vectors(const BatchWriteRequest& request);

    /**
     * @brief Delete a vector from the distributed system
     * @param database_id Database ID
     * @param vector_id Vector ID
     * @param consistency_level Consistency level
     * @return True if successful
     */
    Result<bool> delete_vector(
        const std::string& database_id,
        const std::string& vector_id,
        WriteConsistencyLevel consistency_level = WriteConsistencyLevel::QUORUM
    );

    /**
     * @brief Delete multiple vectors
     * @param database_id Database ID
     * @param vector_ids Vector IDs
     * @param consistency_level Consistency level
     * @return Number of vectors deleted
     */
    Result<int> batch_delete_vectors(
        const std::string& database_id,
        const std::vector<std::string>& vector_ids,
        WriteConsistencyLevel consistency_level = WriteConsistencyLevel::QUORUM
    );

    // ===== Statistics =====

    struct WriteStats {
        int64_t total_writes;
        int64_t successful_writes;
        int64_t failed_writes;
        double success_rate;
        double avg_write_latency_ms;
        double avg_replication_latency_ms;
        int64_t total_replications;
        int64_t successful_replications;
        int64_t failed_replications;
        double replication_success_rate;
        int64_t async_queue_size;
    };

    WriteStats get_statistics() const;
    void reset_statistics();

    // ===== Configuration =====

    CoordinatorConfig get_config() const { return config_; }
    Result<bool> update_config(const CoordinatorConfig& new_config);

private:
    // Write routing

    /**
     * @brief Determine target shard for a vector
     */
    Result<std::string> determine_target_shard(
        const std::string& database_id,
        const std::string& vector_id
    );

    /**
     * @brief Get worker ID for a shard
     */
    Result<std::string> get_shard_worker(const std::string& shard_id);

    /**
     * @brief Get replica workers for a shard
     */
    Result<std::vector<std::string>> get_replica_workers(
        const std::string& shard_id,
        const std::string& primary_worker_id
    );

    // Replication

    /**
     * @brief Perform synchronous replication
     */
    Result<int> replicate_sync(
        const std::string& shard_id,
        const std::string& primary_worker_id,
        const std::vector<std::string>& replica_workers,
        const std::vector<Vector>& vectors,
        WriteConsistencyLevel consistency_level
    );

    /**
     * @brief Enqueue async replication task
     */
    Result<bool> replicate_async(
        const std::string& shard_id,
        const std::string& primary_worker_id,
        const std::vector<std::string>& replica_workers,
        const std::vector<Vector>& vectors
    );

    /**
     * @brief Async replication worker thread
     */
    void async_replication_worker();

    /**
     * @brief Process a replication task
     */
    bool process_replication_task(const ReplicationTask& task);

    /**
     * @brief Replicate vectors to a specific worker
     */
    Result<bool> replicate_to_worker(
        const std::string& worker_id,
        const std::string& shard_id,
        const std::vector<Vector>& vectors
    );

    // Conflict resolution

    /**
     * @brief Resolve write conflicts using configured strategy
     */
    Result<Vector> resolve_conflict(
        const Vector& existing_vector,
        const Vector& new_vector
    );

    /**
     * @brief Last-write-wins conflict resolution
     */
    Vector resolve_lww(const Vector& existing_vector, const Vector& new_vector);

    // Consistency level helpers

    /**
     * @brief Calculate required acknowledgments for consistency level
     */
    int calculate_required_acks(
        WriteConsistencyLevel level,
        int total_replicas
    ) const;

    /**
     * @brief Check if consistency requirements are met
     */
    bool meets_consistency_requirements(
        int acks_received,
        WriteConsistencyLevel level,
        int total_replicas
    ) const;

    // Statistics
    void record_write(bool success, int64_t latency_ms);
    void record_replication(bool success, int64_t latency_ms);
    void update_async_queue_size();

    // Utility
    std::string generate_request_id() const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_DISTRIBUTED_WRITE_COORDINATOR_H
