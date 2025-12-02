#ifndef JADEVECTORDB_DISTRIBUTED_WORKER_SERVICE_H
#define JADEVECTORDB_DISTRIBUTED_WORKER_SERVICE_H

#include "lib/error_handling.h"
#include "lib/logging.h"
#include "models/vector.h"
#include "models/database.h"
#include "services/similarity_search.h"
#include "services/sharding_service.h"
#include "services/cluster_service.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>

#ifdef BUILD_WITH_GRPC
#include <grpcpp/grpcpp.h>
#include "distributed.grpc.pb.h"
#endif

namespace jadevectordb {

// Forward declarations
class VectorDatabase;
class SimilaritySearchService;

/**
 * @brief gRPC service implementation for distributed worker nodes
 *
 * This service handles RPC requests from the master node for:
 * - Shard-based search operations
 * - Write operations to local shards
 * - Health monitoring
 * - Shard management (assignment, removal, sync)
 * - Replication operations
 * - Raft consensus participation
 */
class DistributedWorkerService {
private:
    std::shared_ptr<logging::Logger> logger_;
    std::string node_id_;
    std::string node_host_;
    int node_port_;

    // Core services
    std::shared_ptr<VectorDatabase> vector_db_;
    std::shared_ptr<SimilaritySearchService> search_service_;
    std::shared_ptr<ShardingService> sharding_service_;
    std::shared_ptr<ClusterService> cluster_service_;

    // Local shard management
    mutable std::mutex shard_mutex_;
    std::unordered_map<std::string, ShardInfo> local_shards_;  // shard_id -> ShardInfo

    // Statistics
    mutable std::mutex stats_mutex_;
    int64_t queries_processed_;
    int64_t writes_processed_;
    int64_t total_query_time_ms_;
    int64_t total_write_time_ms_;
    std::chrono::steady_clock::time_point start_time_;

    // Server state
    bool initialized_;
    bool running_;

#ifdef BUILD_WITH_GRPC
    std::unique_ptr<grpc::Server> grpc_server_;
#endif

public:
    DistributedWorkerService(
        const std::string& node_id,
        const std::string& host,
        int port,
        std::shared_ptr<VectorDatabase> vector_db,
        std::shared_ptr<SimilaritySearchService> search_service,
        std::shared_ptr<ShardingService> sharding_service,
        std::shared_ptr<ClusterService> cluster_service
    );

    ~DistributedWorkerService();

    // ===== Lifecycle =====

    Result<bool> initialize();
    Result<bool> start();
    Result<bool> stop();
    bool is_running() const { return running_; }

    // ===== Search Operations =====

    // Execute search on a specific local shard
    Result<SearchResults> execute_shard_search(
        const std::string& shard_id,
        const std::string& request_id,
        const std::vector<float>& query_vector,
        int top_k,
        const std::string& metric_type,
        float threshold,
        const std::unordered_map<std::string, std::string>& filters
    );

    // ===== Write Operations =====

    // Write a single vector to shard
    Result<bool> write_to_shard(
        const std::string& shard_id,
        const std::string& request_id,
        const Vector& vector,
        ConsistencyLevel consistency_level,
        bool wait_for_replication
    );

    // Batch write vectors to shard
    Result<int> batch_write_to_shard(
        const std::string& shard_id,
        const std::string& request_id,
        const std::vector<Vector>& vectors,
        ConsistencyLevel consistency_level,
        bool wait_for_replication
    );

    // Delete vectors from shard
    Result<int> delete_from_shard(
        const std::string& shard_id,
        const std::string& request_id,
        const std::vector<std::string>& vector_ids,
        ConsistencyLevel consistency_level
    );

    // ===== Health & Monitoring =====

    struct HealthInfo {
        HealthStatus status;
        std::string version;
        int64_t uptime_seconds;
        ResourceUsage resource_usage;
        std::vector<ShardStatus> shard_statuses;
    };

    Result<HealthInfo> get_health();

    struct WorkerStatistics {
        int64_t total_vectors;
        int32_t active_shards;
        int64_t queries_processed;
        int64_t writes_processed;
        double avg_query_latency_ms;
        double avg_write_latency_ms;
        ResourceUsage resource_usage;
        std::vector<ShardStats> shard_stats;
    };

    Result<WorkerStatistics> get_worker_stats(bool include_shard_details);

    // ===== Shard Management =====

    // Assign a new shard to this worker
    Result<bool> assign_shard(
        const std::string& shard_id,
        bool is_primary,
        const ShardConfig& config,
        const std::vector<uint8_t>& initial_data
    );

    // Remove a shard from this worker
    Result<bool> remove_shard(
        const std::string& shard_id,
        bool force,
        bool transfer_to_worker,
        const std::string& target_worker_id
    );

    // Get information about a specific shard
    Result<ShardInfo> get_shard_info(
        const std::string& shard_id,
        bool include_statistics
    );

    // ===== Replication Operations =====

    struct ReplicationResult {
        int32_t vectors_replicated;
        int64_t replication_lag_ms;
        int64_t current_version;
    };

    // Replicate data to this node
    Result<ReplicationResult> replicate_data(
        const std::string& shard_id,
        const std::string& source_node_id,
        ReplicationType replication_type,
        const std::vector<Vector>& vectors,
        int64_t from_version,
        int64_t to_version
    );

    // Synchronize shard data with another node
    Result<bool> sync_shard(
        const std::string& shard_id,
        const std::string& source_node_id,
        int64_t target_version
    );

    // ===== Raft Consensus Operations =====

    struct VoteResult {
        int64_t term;
        bool vote_granted;
    };

    Result<VoteResult> handle_vote_request(
        int64_t term,
        const std::string& candidate_id,
        int64_t last_log_index,
        int64_t last_log_term
    );

    struct HeartbeatResult {
        int64_t term;
        bool success;
        int64_t match_index;
    };

    Result<HeartbeatResult> handle_heartbeat(
        int64_t term,
        const std::string& leader_id,
        int64_t prev_log_index,
        int64_t prev_log_term,
        int64_t leader_commit_index
    );

    struct AppendEntriesResult {
        int64_t term;
        bool success;
        int64_t match_index;
    };

    Result<AppendEntriesResult> handle_append_entries(
        int64_t term,
        const std::string& leader_id,
        int64_t prev_log_index,
        int64_t prev_log_term,
        const std::vector<LogEntry>& entries,
        int64_t leader_commit_index
    );

private:
    // Helper methods
    Result<bool> validate_shard_exists(const std::string& shard_id) const;
    Result<bool> validate_shard_is_active(const std::string& shard_id) const;
    Result<ResourceUsage> collect_resource_usage() const;
    Result<std::vector<ShardStatus>> collect_shard_statuses() const;
    Result<std::vector<ShardStats>> collect_shard_stats() const;
    void record_query_time(int64_t time_ms);
    void record_write_time(int64_t time_ms);

    // Shard data management
    Result<bool> load_shard_data(const std::string& shard_id, const std::vector<uint8_t>& data);
    Result<std::vector<uint8_t>> export_shard_data(const std::string& shard_id) const;
};

#ifdef BUILD_WITH_GRPC

/**
 * @brief gRPC service implementation wrapper
 *
 * Wraps DistributedWorkerService to provide gRPC interface
 */
class DistributedWorkerServiceImpl final : public distributed::DistributedService::Service {
private:
    std::shared_ptr<DistributedWorkerService> worker_service_;
    std::shared_ptr<logging::Logger> logger_;

public:
    explicit DistributedWorkerServiceImpl(std::shared_ptr<DistributedWorkerService> worker_service);

    // Search operations
    grpc::Status ExecuteShardSearch(
        grpc::ServerContext* context,
        const distributed::ShardSearchRequest* request,
        distributed::ShardSearchResponse* response) override;

    // Write operations
    grpc::Status WriteToShard(
        grpc::ServerContext* context,
        const distributed::ShardWriteRequest* request,
        distributed::ShardWriteResponse* response) override;

    grpc::Status BatchWriteToShard(
        grpc::ServerContext* context,
        const distributed::BatchShardWriteRequest* request,
        distributed::ShardWriteResponse* response) override;

    grpc::Status DeleteFromShard(
        grpc::ServerContext* context,
        const distributed::ShardDeleteRequest* request,
        distributed::ShardDeleteResponse* response) override;

    // Health & monitoring
    grpc::Status HealthCheck(
        grpc::ServerContext* context,
        const distributed::HealthCheckRequest* request,
        distributed::HealthCheckResponse* response) override;

    grpc::Status GetWorkerStats(
        grpc::ServerContext* context,
        const distributed::WorkerStatsRequest* request,
        distributed::WorkerStatsResponse* response) override;

    // Shard management
    grpc::Status AssignShard(
        grpc::ServerContext* context,
        const distributed::AssignShardRequest* request,
        distributed::AssignShardResponse* response) override;

    grpc::Status RemoveShard(
        grpc::ServerContext* context,
        const distributed::RemoveShardRequest* request,
        distributed::RemoveShardResponse* response) override;

    grpc::Status GetShardInfo(
        grpc::ServerContext* context,
        const distributed::ShardInfoRequest* request,
        distributed::ShardInfoResponse* response) override;

    // Replication operations
    grpc::Status ReplicateData(
        grpc::ServerContext* context,
        const distributed::ReplicationRequest* request,
        distributed::ReplicationResponse* response) override;

    grpc::Status SyncShard(
        grpc::ServerContext* context,
        const distributed::ShardSyncRequest* request,
        distributed::ShardSyncResponse* response) override;

    // Raft consensus
    grpc::Status RequestVote(
        grpc::ServerContext* context,
        const distributed::VoteRequest* request,
        distributed::VoteResponse* response) override;

    grpc::Status SendHeartbeat(
        grpc::ServerContext* context,
        const distributed::HeartbeatRequest* request,
        distributed::HeartbeatResponse* response) override;

    grpc::Status AppendEntries(
        grpc::ServerContext* context,
        const distributed::AppendEntriesRequest* request,
        distributed::AppendEntriesResponse* response) override;

private:
    // Helper conversion methods
    SearchResults convert_to_search_results(const distributed::ShardSearchResponse& response);
    void populate_search_response(const SearchResults& results, distributed::ShardSearchResponse* response);
};

#endif // BUILD_WITH_GRPC

} // namespace jadevectordb

#endif // JADEVECTORDB_DISTRIBUTED_WORKER_SERVICE_H
