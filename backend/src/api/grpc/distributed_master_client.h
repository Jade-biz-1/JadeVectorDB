#ifndef JADEVECTORDB_DISTRIBUTED_MASTER_CLIENT_H
#define JADEVECTORDB_DISTRIBUTED_MASTER_CLIENT_H

#include "lib/error_handling.h"
#include "lib/logging.h"
#include "models/vector.h"
#include "models/database.h"
#include "similarity_search.h"
#include "distributed_types.h"
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
struct ShardInfo;

/**
 * @brief gRPC client for master node to communicate with worker nodes
 *
 * This client provides methods for the master to:
 * - Execute distributed search queries across workers
 * - Distribute write operations to appropriate shards
 * - Monitor worker health
 * - Manage shard assignments
 * - Coordinate replication
 * - Participate in Raft consensus
 */
class DistributedMasterClient {
public:
    // Configuration for RPC calls
    struct RpcConfig {
        std::chrono::milliseconds default_timeout{5000};
        std::chrono::milliseconds search_timeout{2000};
        std::chrono::milliseconds write_timeout{3000};
        std::chrono::milliseconds health_check_timeout{1000};
        int max_retries{3};
        std::chrono::milliseconds retry_backoff_base{100};
        bool enable_compression{true};

        RpcConfig() = default;
    };

    // Connection information for a worker node
    struct WorkerConnection {
        std::string worker_id;
        std::string host;
        int port;
        bool is_active;
        std::chrono::steady_clock::time_point last_success;
        int consecutive_failures;

#ifdef BUILD_WITH_GRPC
        std::shared_ptr<grpc::Channel> channel;
        std::unique_ptr<distributed::DistributedService::Stub> stub;
#endif
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    RpcConfig config_;

    // Worker connections
    mutable std::mutex connections_mutex_;
    std::unordered_map<std::string, std::shared_ptr<WorkerConnection>> worker_connections_;

    // Statistics
    mutable std::mutex stats_mutex_;
    int64_t total_requests_;
    int64_t failed_requests_;
    int64_t total_request_time_ms_;

    bool initialized_;

public:
    explicit DistributedMasterClient(const RpcConfig& config);
    DistributedMasterClient();  // Default constructor
    ~DistributedMasterClient();

    // ===== Lifecycle =====

    Result<bool> initialize();
    Result<bool> shutdown();
    bool is_initialized() const { return initialized_; }

    // ===== Connection Management =====

    // Add a worker connection
    Result<bool> add_worker(
        const std::string& worker_id,
        const std::string& host,
        int port
    );

    // Remove a worker connection
    Result<bool> remove_worker(const std::string& worker_id);

    // Get all connected workers
    std::vector<std::string> get_connected_workers() const;

    // Check if worker is connected
    bool is_worker_connected(const std::string& worker_id) const;

    // ===== Search Operations =====

    struct SearchRequest {
        std::string shard_id;
        std::string request_id;
        std::vector<float> query_vector;
        int top_k;
        std::string metric_type;
        float threshold;
        std::unordered_map<std::string, std::string> filters;
        std::chrono::milliseconds timeout;
    };

    struct SearchResponse {
        std::string shard_id;
        std::string request_id;
        std::vector<SearchResult> results;
        bool success;
        std::string error_message;
        int64_t execution_time_ms;
        int32_t vectors_scanned;
    };

    // Execute search on a specific worker's shard
    Result<SearchResponse> execute_shard_search(
        const std::string& worker_id,
        const SearchRequest& request
    );

    // Execute search across multiple workers in parallel
    Result<std::vector<SearchResponse>> execute_distributed_search(
        const std::vector<std::string>& worker_ids,
        const SearchRequest& request
    );

    // ===== Write Operations =====

    struct WriteRequest {
        std::string shard_id;
        std::string request_id;
        Vector vector;
        ConsistencyLevel consistency_level;
        bool wait_for_replication;
    };

    struct WriteResponse {
        std::string shard_id;
        bool success;
        std::string error_message;
        int64_t execution_time_ms;
    };

    // Write vector to a specific worker's shard
    Result<WriteResponse> write_to_shard(
        const std::string& worker_id,
        const WriteRequest& request
    );

    // Batch write vectors to a worker's shard
    Result<WriteResponse> batch_write_to_shard(
        const std::string& worker_id,
        const std::string& shard_id,
        const std::string& request_id,
        const std::vector<Vector>& vectors,
        ConsistencyLevel consistency_level
    );

    // Delete vectors from a worker's shard
    Result<bool> delete_from_shard(
        const std::string& worker_id,
        const std::string& shard_id,
        const std::string& request_id,
        const std::vector<std::string>& vector_ids,
        ConsistencyLevel consistency_level
    );

    // ===== Health & Monitoring =====

    struct HealthCheckResponse {
        std::string worker_id;
        HealthStatus status;
        std::string version;
        int64_t uptime_seconds;
        ResourceUsage resource_usage;
        std::vector<ShardStatus> shard_statuses;
        bool success;
    };

    // Check health of a specific worker
    Result<HealthCheckResponse> check_worker_health(const std::string& worker_id);

    // Check health of all workers
    Result<std::unordered_map<std::string, HealthCheckResponse>> check_all_workers_health();

    struct WorkerStatsResponse {
        std::string worker_id;
        int64_t total_vectors;
        int32_t active_shards;
        int64_t queries_processed;
        int64_t writes_processed;
        double avg_query_latency_ms;
        double avg_write_latency_ms;
        ResourceUsage resource_usage;
        std::vector<ShardStats> shard_stats;
    };

    // Get statistics from a worker
    Result<WorkerStatsResponse> get_worker_stats(
        const std::string& worker_id,
        bool include_shard_details = false
    );

    // ===== Shard Management =====

    // Assign a shard to a worker
    Result<bool> assign_shard(
        const std::string& worker_id,
        const std::string& shard_id,
        bool is_primary,
        const ShardConfig& config,
        const std::vector<uint8_t>& initial_data = {}
    );

    // Remove a shard from a worker
    Result<bool> remove_shard(
        const std::string& worker_id,
        const std::string& shard_id,
        bool force = false,
        const std::string& target_worker_id = ""
    );

    // Get shard information from a worker
    Result<ShardInfo> get_shard_info(
        const std::string& worker_id,
        const std::string& shard_id,
        bool include_statistics = false
    );

    // ===== Replication Operations =====

    struct ReplicationRequest {
        std::string shard_id;
        std::string source_node_id;
        ReplicationType replication_type;
        std::vector<Vector> vectors;
        int64_t from_version;
        int64_t to_version;
    };

    struct ReplicationResponse {
        std::string shard_id;
        bool success;
        int32_t vectors_replicated;
        int64_t replication_lag_ms;
        int64_t current_version;
    };

    // Replicate data to a worker
    Result<ReplicationResponse> replicate_data(
        const std::string& target_worker_id,
        const ReplicationRequest& request
    );

    // Synchronize shard between workers
    Result<bool> sync_shard(
        const std::string& target_worker_id,
        const std::string& shard_id,
        const std::string& source_worker_id,
        int64_t target_version
    );

    // ===== Raft Consensus Operations =====

    struct VoteRequest {
        int64_t term;
        std::string candidate_id;
        int64_t last_log_index;
        int64_t last_log_term;
    };

    struct VoteResponse {
        int64_t term;
        bool vote_granted;
        std::string voter_id;
    };

    // Request vote from a worker
    Result<VoteResponse> request_vote(
        const std::string& worker_id,
        const VoteRequest& request
    );

    struct HeartbeatRequest {
        int64_t term;
        std::string leader_id;
        int64_t prev_log_index;
        int64_t prev_log_term;
        int64_t leader_commit_index;
    };

    struct HeartbeatResponse {
        int64_t term;
        bool success;
        std::string follower_id;
        int64_t match_index;
    };

    // Send heartbeat to a worker
    Result<HeartbeatResponse> send_heartbeat(
        const std::string& worker_id,
        const HeartbeatRequest& request
    );

    struct AppendEntriesRequest {
        int64_t term;
        std::string leader_id;
        int64_t prev_log_index;
        int64_t prev_log_term;
        std::vector<LogEntry> entries;
        int64_t leader_commit_index;
    };

    struct AppendEntriesResponse {
        int64_t term;
        bool success;
        std::string follower_id;
        int64_t match_index;
    };

    // Append entries to a worker
    Result<AppendEntriesResponse> append_entries(
        const std::string& worker_id,
        const AppendEntriesRequest& request
    );

    struct InstallSnapshotRequest {
        int64_t term;
        std::string leader_id;
        int64_t last_included_index;
        int64_t last_included_term;
        int64_t offset;
        std::vector<uint8_t> data;
        bool done;
    };

    struct InstallSnapshotResponse {
        int64_t term;
        bool success;
        std::string follower_id;
    };

    // Install snapshot to a worker
    Result<InstallSnapshotResponse> install_snapshot(
        const std::string& worker_id,
        const InstallSnapshotRequest& request
    );

    // ===== Statistics & Monitoring =====

    struct ClientStatistics {
        int64_t total_requests;
        int64_t failed_requests;
        double failure_rate;
        double avg_request_time_ms;
        int active_connections;
    };

    ClientStatistics get_statistics() const;

    // Reset statistics
    void reset_statistics();

private:
    // Helper methods

    // Get or create worker connection
    Result<std::shared_ptr<WorkerConnection>> get_worker_connection(const std::string& worker_id);

    // Create gRPC channel for worker
#ifdef BUILD_WITH_GRPC
    std::shared_ptr<grpc::Channel> create_channel(const std::string& host, int port);

    // Create client context with timeout
    std::unique_ptr<grpc::ClientContext> create_context(std::chrono::milliseconds timeout);
#endif

    // Record request statistics
    void record_request(bool success, int64_t duration_ms);

    // Mark worker as failed
    void mark_worker_failed(const std::string& worker_id);

    // Mark worker as successful
    void mark_worker_success(const std::string& worker_id);

    // Retry logic with exponential backoff
    template<typename Func>
    Result<typename std::invoke_result<Func>::type> retry_with_backoff(
        Func&& func,
        int max_attempts
    );
};

} // namespace jadevectordb

#endif // JADEVECTORDB_DISTRIBUTED_MASTER_CLIENT_H
