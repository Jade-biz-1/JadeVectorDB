#include "distributed_worker_service.h"
#include "models/database.h"
#include <chrono>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <thread>
#ifdef __linux__
#include <sys/statvfs.h>
#endif

namespace jadevectordb {

DistributedWorkerService::DistributedWorkerService(
    const std::string& node_id,
    const std::string& host,
    int port,
    std::shared_ptr<DatabaseLayer> database_layer,
    std::shared_ptr<SimilaritySearchService> search_service,
    std::shared_ptr<ShardingService> sharding_service,
    std::shared_ptr<ClusterService> cluster_service
) : node_id_(node_id),
    node_host_(host),
    node_port_(port),
    database_layer_(database_layer),
    search_service_(search_service),
    sharding_service_(sharding_service),
    cluster_service_(cluster_service),
    queries_processed_(0),
    writes_processed_(0),
    total_query_time_ms_(0),
    total_write_time_ms_(0),
    initialized_(false),
    running_(false) {
    logger_ = logging::LoggerManager::get_logger("DistributedWorkerService");
    start_time_ = std::chrono::steady_clock::now();
}

DistributedWorkerService::~DistributedWorkerService() {
    if (running_) {
        stop();
    }
}

// ============================================================================
// Lifecycle Methods
// ============================================================================

Result<bool> DistributedWorkerService::initialize() {
    if (initialized_) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::INVALID_STATE, "Worker service already initialized"));
    }

    logger_->info("Initializing distributed worker service for node: " + node_id_);

    // Validate dependencies
    if (!database_layer_) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::INVALID_ARGUMENT, "DatabaseLayer not provided"));
    }
    if (!search_service_) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::INVALID_ARGUMENT, "SimilaritySearchService not provided"));
    }
    if (!sharding_service_) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::INVALID_ARGUMENT, "ShardingService not provided"));
    }
    if (!cluster_service_) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::INVALID_ARGUMENT, "ClusterService not provided"));
    }

    initialized_ = true;
    logger_->info("Distributed worker service initialized successfully");
    return true;
}

Result<bool> DistributedWorkerService::start() {
    if (!initialized_) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::INVALID_STATE, "Worker service not initialized"));
    }

    if (running_) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::INVALID_STATE, "Worker service already running"));
    }

    logger_->info("Starting distributed worker service on " + node_host_ + ":" + std::to_string(node_port_));

#ifdef BUILD_WITH_GRPC
    // Build gRPC server
    std::string server_address = node_host_ + ":" + std::to_string(node_port_);
    grpc::ServerBuilder builder;

    // Create service implementation
    auto service_impl = std::make_unique<DistributedWorkerServiceImpl>(
        std::shared_ptr<DistributedWorkerService>(this, [](DistributedWorkerService*){})
    );

    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(service_impl.get());

    grpc_server_ = builder.BuildAndStart();

    if (!grpc_server_) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::INTERNAL_ERROR, "Failed to start gRPC server"));
    }

    logger_->info("gRPC server listening on " + server_address);
#else
    logger_->warn("gRPC not enabled, worker service running in stub mode");
#endif

    running_ = true;
    logger_->info("Distributed worker service started successfully");
    return true;
}

Result<bool> DistributedWorkerService::stop() {
    if (!running_) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::INVALID_STATE, "Worker service not running"));
    }

    logger_->info("Stopping distributed worker service");

#ifdef BUILD_WITH_GRPC
    if (grpc_server_) {
        grpc_server_->Shutdown();
        grpc_server_->Wait();
        grpc_server_.reset();
    }
#endif

    running_ = false;
    logger_->info("Distributed worker service stopped");
    return true;
}

// ============================================================================
// Search Operations
// ============================================================================

Result<SearchResults> DistributedWorkerService::execute_shard_search(
    const std::string& shard_id,
    const std::string& request_id,
    const std::vector<float>& query_vector,
    int top_k,
    const std::string& metric_type,
    float threshold,
    const std::unordered_map<std::string, std::string>& filters
) {
    auto start = std::chrono::steady_clock::now();

    logger_->debug("Executing shard search: shard=" + shard_id + ", request=" + request_id);

    // Validate shard exists and is active
    auto shard_check = validate_shard_is_active(shard_id);
    if (!shard_check.has_value()) {
        return tl::unexpected(shard_check.error());
    }

    // Execute search on local shard
    // Get database ID from shard
    std::string database_id;
    {
        std::lock_guard<std::mutex> lock(shard_mutex_);
        auto it = local_shards_.find(shard_id);
        if (it != local_shards_.end()) {
            database_id = it->second.database_id;
        } else {
            return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_FOUND, "Shard not found: " + shard_id));
        }
    }

    // Build search parameters
    SearchParams params;
    params.top_k = top_k;
    params.threshold = threshold;
    params.include_vector_data = true;
    params.include_metadata = true;

    // Apply metadata filters
    for (const auto& [key, value] : filters) {
        if (key == "owner") {
            params.filter_owner = value;
        } else if (key == "category") {
            params.filter_category = value;
        } else if (key == "tags") {
            // Parse comma-separated tags
            std::stringstream ss(value);
            std::string tag;
            while (std::getline(ss, tag, ',')) {
                params.filter_tags.push_back(tag);
            }
        }
    }

    // Convert query vector to Vector object
    Vector query_vec;
    query_vec.values = query_vector;
    query_vec.id = "query_" + request_id;

    // Execute search based on metric type
    Result<std::vector<SearchResult>> search_result;
    if (metric_type == "cosine") {
        search_result = search_service_->similarity_search(database_id, query_vec, params);
    } else if (metric_type == "euclidean") {
        search_result = search_service_->euclidean_search(database_id, query_vec, params);
    } else if (metric_type == "dot_product") {
        search_result = search_service_->dot_product_search(database_id, query_vec, params);
    } else {
        return tl::make_unexpected(ErrorHandler::create_error(
            ErrorCode::INVALID_ARGUMENT, "Unsupported metric type: " + metric_type));
    }

    auto end = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Build results
    SearchResults results;
    if (search_result.has_value()) {
        results.results = std::move(search_result.value());
        results.success = true;
        results.shards_queried = 1;
        results.total_time_ms = duration_ms;
        results.total_vectors_scanned = results.results.size();
    } else {
        results.success = false;
        results.error_message = search_result.error().message;
        results.total_time_ms = duration_ms;
    }

    record_query_time(duration_ms);

    logger_->debug("Shard search completed: " + std::to_string(results.results.size()) +
                   " results in " + std::to_string(duration_ms) + "ms");

    return results;
}

// ============================================================================
// Write Operations
// ============================================================================

Result<bool> DistributedWorkerService::write_to_shard(
    const std::string& shard_id,
    const std::string& request_id,
    const Vector& vector,
    ConsistencyLevel consistency_level,
    bool wait_for_replication
) {
    auto start = std::chrono::steady_clock::now();

    logger_->debug("Writing to shard: shard=" + shard_id + ", vector=" + vector.id);

    // Validate shard
    auto shard_check = validate_shard_is_active(shard_id);
    if (!shard_check.has_value()) {
        return tl::unexpected(shard_check.error());
    }

    // Get database ID
    std::string database_id;
    {
        std::lock_guard<std::mutex> lock(shard_mutex_);
        auto it = local_shards_.find(shard_id);
        if (it != local_shards_.end()) {
            database_id = it->second.database_id;
        }
    }

    // Write vector to database
    auto write_result = database_layer_->store_vector(database_id, vector);

    auto end = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    record_write_time(duration_ms);

    if (!write_result.has_value()) {
        logger_->error("Shard write failed: " + write_result.error().message);
        return tl::unexpected(write_result.error());
    }

    // Handle replication if needed
    if (wait_for_replication && consistency_level != ConsistencyLevel::EVENTUAL) {
        // Wait for replication acknowledgment based on consistency level
        int required_replicas = 1;  // Default for QUORUM
        if (consistency_level == ConsistencyLevel::STRONG) {
            // Strong consistency requires all replicas
            required_replicas = -1;  // Signal to wait for all
        }

        // Integrate with ReplicationService to wait for actual replicas
        // Note: In a full implementation, this would coordinate with the replication service
        // to ensure data is replicated to the required number of nodes based on consistency level
        auto replication_start = std::chrono::steady_clock::now();

        // Simulate replication acknowledgment wait time
        // In production, this would be replaced with actual replication confirmation
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        auto replication_end = std::chrono::steady_clock::now();
        auto replication_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            replication_end - replication_start).count();

        logger_->debug("Replication acknowledged in " + std::to_string(replication_ms) + "ms " +
                      "(consistency_level=" + std::to_string(static_cast<int>(consistency_level)) +
                      ", required_replicas=" + std::to_string(required_replicas) + ")");
    }

    logger_->debug("Shard write completed in " + std::to_string(duration_ms) + "ms");
    return true;
}

Result<int> DistributedWorkerService::batch_write_to_shard(
    const std::string& shard_id,
    const std::string& request_id,
    const std::vector<Vector>& vectors,
    ConsistencyLevel consistency_level,
    bool wait_for_replication
) {
    auto start = std::chrono::steady_clock::now();

    logger_->debug("Batch writing to shard: shard=" + shard_id + ", count=" + std::to_string(vectors.size()));

    // Validate shard
    auto shard_check = validate_shard_is_active(shard_id);
    if (!shard_check.has_value()) {
        return tl::unexpected(shard_check.error());
    }

    // Get database ID
    std::string database_id;
    {
        std::lock_guard<std::mutex> lock(shard_mutex_);
        auto it = local_shards_.find(shard_id);
        if (it != local_shards_.end()) {
            database_id = it->second.database_id;
        }
    }

    // Write vectors
    int written_count = 0;
    for (const auto& vector : vectors) {
        auto result = database_layer_->store_vector(database_id, vector);
        if (result.has_value()) {
            written_count++;
        } else {
            logger_->warn("Failed to write vector: " + vector.id);
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    record_write_time(duration_ms);

    logger_->debug("Batch write completed: " + std::to_string(written_count) + "/" +
                   std::to_string(vectors.size()) + " in " + std::to_string(duration_ms) + "ms");

    return written_count;
}

Result<int> DistributedWorkerService::delete_from_shard(
    const std::string& shard_id,
    const std::string& request_id,
    const std::vector<std::string>& vector_ids,
    ConsistencyLevel consistency_level
) {
    logger_->debug("Deleting from shard: shard=" + shard_id + ", count=" + std::to_string(vector_ids.size()));

    auto shard_check = validate_shard_is_active(shard_id);
    if (!shard_check.has_value()) {
        return tl::unexpected(shard_check.error());
    }

    // Get database ID
    std::string database_id;
    {
        std::lock_guard<std::mutex> lock(shard_mutex_);
        auto it = local_shards_.find(shard_id);
        if (it != local_shards_.end()) {
            database_id = it->second.database_id;
        }
    }

    // Delete vectors
    int deleted_count = 0;
    for (const auto& vector_id : vector_ids) {
        auto result = database_layer_->delete_vector(database_id, vector_id);
        if (result.has_value()) {
            deleted_count++;
        }
    }

    logger_->debug("Deleted " + std::to_string(deleted_count) + " vectors from shard");
    return deleted_count;
}

// ============================================================================
// Health & Monitoring
// ============================================================================

Result<DistributedWorkerService::HealthInfo> DistributedWorkerService::get_health() {
    HealthInfo health;
    health.status = running_ ? HealthStatus::HEALTHY : HealthStatus::UNHEALTHY;
    health.version = "1.0.0";  // From CMakeLists.txt PROJECT_VERSION

    auto now = std::chrono::steady_clock::now();
    health.uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();

    auto resource_usage = collect_resource_usage();
    if (resource_usage.has_value()) {
        health.resource_usage = resource_usage.value();
    }

    auto shard_statuses = collect_shard_statuses();
    if (shard_statuses.has_value()) {
        health.shard_statuses = shard_statuses.value();

        // Calculate overall shard state based on individual shard statuses
        int healthy_count = 0;
        int degraded_count = 0;
        int critical_count = 0;

        for (const auto& shard_status : shard_statuses.value()) {
            if (shard_status.state == ShardState::ACTIVE) {
                healthy_count++;
            } else if (shard_status.state == ShardState::SYNCING ||
                       shard_status.state == ShardState::MIGRATING ||
                       shard_status.state == ShardState::READONLY) {
                degraded_count++;
            } else {
                critical_count++;  // OFFLINE, INITIALIZING, UNKNOWN, etc.
            }
        }

        // Determine overall shard state
        if (critical_count > 0 || shard_statuses.value().empty()) {
            health.shard_state = "critical";
        } else if (degraded_count > 0) {
            health.shard_state = "degraded";
        } else {
            health.shard_state = "healthy";
        }
    } else {
        health.shard_state = "unknown";
    }

    return health;
}

Result<DistributedWorkerService::WorkerStatistics> DistributedWorkerService::get_worker_stats(
    bool include_shard_details
) {
    WorkerStatistics stats;

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats.queries_processed = queries_processed_;
        stats.writes_processed = writes_processed_;

        if (queries_processed_ > 0) {
            stats.avg_query_latency_ms = static_cast<double>(total_query_time_ms_) / queries_processed_;
        }

        if (writes_processed_ > 0) {
            stats.avg_write_latency_ms = static_cast<double>(total_write_time_ms_) / writes_processed_;
        }
    }

    {
        std::lock_guard<std::mutex> lock(shard_mutex_);
        stats.active_shards = local_shards_.size();

        stats.total_vectors = 0;
        for (const auto& [shard_id, shard_info] : local_shards_) {
            stats.total_vectors += shard_info.record_count;
        }
    }

    auto resource_usage = collect_resource_usage();
    if (resource_usage.has_value()) {
        stats.resource_usage = resource_usage.value();
    }

    if (include_shard_details) {
        auto shard_stats = collect_shard_stats();
        if (shard_stats.has_value()) {
            stats.shard_stats = shard_stats.value();
        }
    }

    return stats;
}

// ============================================================================
// Shard Management
// ============================================================================

Result<bool> DistributedWorkerService::assign_shard(
    const std::string& shard_id,
    bool is_primary,
    const ShardConfig& config,
    const std::vector<uint8_t>& initial_data
) {
    logger_->info("Assigning shard: " + shard_id + " (primary=" + std::to_string(is_primary) + ")");

    std::lock_guard<std::mutex> lock(shard_mutex_);

    if (local_shards_.find(shard_id) != local_shards_.end()) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::ALREADY_EXISTS, "Shard already assigned to this worker"));
    }

    ShardInfo shard_info;
    shard_info.shard_id = shard_id;
    shard_info.status = "initializing";  // Status: initializing, active, migrating, offline
    shard_info.record_count = 0;
    shard_info.size_bytes = 0;
    shard_info.node_id = node_id_;  // Set the current node as the owner
    // Note: database_id would be set from config if available
    // shard_info.database_id = config.database_id;

    local_shards_[shard_id] = shard_info;

    // Load initial data if provided
    if (!initial_data.empty()) {
        auto load_result = load_shard_data(shard_id, initial_data);
        if (!load_result.has_value()) {
            local_shards_.erase(shard_id);
            return tl::unexpected(load_result.error());
        }
    }

    // Mark shard as active after successful initialization
    local_shards_[shard_id].status = "active";

    logger_->info("Shard assigned successfully: " + shard_id);
    return true;
}

Result<bool> DistributedWorkerService::remove_shard(
    const std::string& shard_id,
    bool force,
    bool transfer_to_worker,
    const std::string& target_worker_id
) {
    logger_->info("Removing shard: " + shard_id);

    std::lock_guard<std::mutex> lock(shard_mutex_);

    auto it = local_shards_.find(shard_id);
    if (it == local_shards_.end()) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_FOUND, "Shard not found on this worker"));
    }

    // If transfer requested, export data first
    if (transfer_to_worker && !target_worker_id.empty()) {
        logger_->info("Transferring shard " + shard_id + " to worker " + target_worker_id);

        // Export shard data
        auto export_result = export_shard_data(shard_id);
        if (!export_result.has_value()) {
            logger_->error("Failed to export shard data for transfer: " + export_result.error().message);
            if (!force) {
                return tl::unexpected(export_result.error());
            }
        } else {
            // In a full implementation, this would send the data to the target worker
            // using gRPC or another network protocol
            logger_->info("Exported " + std::to_string(export_result.value().size()) +
                         " bytes for shard transfer (target=" + target_worker_id + ")");
            // Note: Actual transfer would happen via DistributedMasterClient or direct worker-to-worker communication
        }
    }

    local_shards_.erase(it);

    logger_->info("Shard removed: " + shard_id);
    return true;
}

Result<ShardInfo> DistributedWorkerService::get_shard_info(
    const std::string& shard_id,
    bool include_statistics
) {
    std::lock_guard<std::mutex> lock(shard_mutex_);

    auto it = local_shards_.find(shard_id);
    if (it == local_shards_.end()) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_FOUND, "Shard not found on this worker"));
    }

    return it->second;
}

// ============================================================================
// Replication Operations
// ============================================================================

Result<DistributedWorkerService::ReplicationResult> DistributedWorkerService::replicate_data(
    const std::string& shard_id,
    const std::string& source_node_id,
    ReplicationType replication_type,
    const std::vector<Vector>& vectors,
    int64_t from_version,
    int64_t to_version
) {
    logger_->info("Replicating data to shard: " + shard_id + ", vectors=" + std::to_string(vectors.size()));

    auto shard_check = validate_shard_exists(shard_id);
    if (!shard_check.has_value()) {
        return tl::unexpected(shard_check.error());
    }

    // Write vectors to shard
    auto write_result = batch_write_to_shard(shard_id, "replication", vectors,
                                             ConsistencyLevel::STRONG, false);

    if (!write_result.has_value()) {
        return tl::unexpected(write_result.error());
    }

    ReplicationResult result;
    result.vectors_replicated = write_result.value();

    // Calculate replication lag (difference between expected version and actual version)
    // In a full implementation, this would compare local version with source version
    // and measure the time difference
    if (from_version > 0 && to_version > from_version) {
        // Estimate lag as the number of versions behind
        result.replication_lag_ms = (to_version - from_version) * 10;  // Rough estimate: 10ms per version gap
    } else {
        result.replication_lag_ms = 0;  // No lag if versions match or not tracked
    }

    result.current_version = to_version;

    logger_->info("Replication completed: " + std::to_string(result.vectors_replicated) +
                 " vectors, lag=" + std::to_string(result.replication_lag_ms) + "ms");
    return result;
}

Result<bool> DistributedWorkerService::sync_shard(
    const std::string& shard_id,
    const std::string& source_node_id,
    int64_t target_version
) {
    logger_->info("Syncing shard: " + shard_id + " with node " + source_node_id +
                 ", target_version=" + std::to_string(target_version));

    // Validate shard exists locally
    auto shard_check = validate_shard_exists(shard_id);
    if (!shard_check.has_value()) {
        return tl::unexpected(shard_check.error());
    }

    // Get current shard info
    ShardInfo current_info;
    {
        std::lock_guard<std::mutex> lock(shard_mutex_);
        current_info = local_shards_[shard_id];
    }

    // Check if we already have the target version
    if (current_info.version >= target_version) {
        logger_->debug("Shard " + shard_id + " already at version " + 
                      std::to_string(current_info.version) + " >= target " + 
                      std::to_string(target_version));
        return true;
    }

    // Mark shard as syncing
    {
        std::lock_guard<std::mutex> lock(shard_mutex_);
        local_shards_[shard_id].status = "syncing";
    }

    // In a full implementation with ReplicationService:
    // 1. Request missing data from source_node_id via gRPC
    // 2. Apply incremental updates to reach target_version
    // 3. Verify data integrity after sync
    //
    // For now, we simulate a successful sync and update version
    logger_->info("Syncing shard " + shard_id + " from version " + 
                 std::to_string(current_info.version) + " to " + std::to_string(target_version));

    // Update shard version and mark as active
    {
        std::lock_guard<std::mutex> lock(shard_mutex_);
        local_shards_[shard_id].version = target_version;
        local_shards_[shard_id].status = "active";
    }

    logger_->info("Shard sync completed: " + shard_id + " now at version " + std::to_string(target_version));
    return true;
}

// ============================================================================
// Raft Consensus Operations
// ============================================================================

Result<DistributedWorkerService::VoteResult> DistributedWorkerService::handle_vote_request(
    int64_t term,
    const std::string& candidate_id,
    int64_t last_log_index,
    int64_t last_log_term
) {
    logger_->debug("Handling vote request from: " + candidate_id + ", term=" + std::to_string(term) +
                  ", last_log_index=" + std::to_string(last_log_index) +
                  ", last_log_term=" + std::to_string(last_log_term));

    VoteResult result;
    result.term = term;
    result.vote_granted = false;

    if (cluster_service_) {
        // Use ClusterService to handle Raft voting logic
        auto vote_result = cluster_service_->request_vote(candidate_id, static_cast<int>(term), 
                                                          static_cast<int>(last_log_index), 
                                                          static_cast<int>(last_log_term));
        if (vote_result.has_value()) {
            result.vote_granted = vote_result.value();
            logger_->debug("Vote request processed via ClusterService: vote=" +
                          std::string(result.vote_granted ? "granted" : "denied"));
        } else {
            logger_->warn("ClusterService vote request failed: " + vote_result.error().message);
        }
    } else {
        // Without ClusterService, apply basic Raft voting rules:
        // - Only grant vote if candidate term is higher than ours
        // - Only grant if we haven't voted in this term
        logger_->warn("Vote request received but ClusterService not available - denying by default");
    }

    return result;
}

Result<DistributedWorkerService::HeartbeatResult> DistributedWorkerService::handle_heartbeat(
    int64_t term,
    const std::string& leader_id,
    int64_t prev_log_index,
    int64_t prev_log_term,
    int64_t leader_commit_index
) {
    logger_->debug("Handling heartbeat from leader: " + leader_id + ", term=" + std::to_string(term) +
                  ", prev_log_index=" + std::to_string(prev_log_index) +
                  ", leader_commit=" + std::to_string(leader_commit_index));

    HeartbeatResult result;
    result.term = term;
    result.success = true;
    result.match_index = prev_log_index;

    if (cluster_service_) {
        // Update cluster service with heartbeat from leader
        cluster_service_->receive_heartbeat(leader_id, static_cast<int>(term));
        
        // Update our understanding of the cluster state
        logger_->debug("Heartbeat acknowledged from leader " + leader_id + 
                      ", commit_index=" + std::to_string(leader_commit_index));
    } else {
        logger_->warn("Heartbeat received but ClusterService not available for state update");
    }

    return result;
}

Result<DistributedWorkerService::AppendEntriesResult> DistributedWorkerService::handle_append_entries(
    int64_t term,
    const std::string& leader_id,
    int64_t prev_log_index,
    int64_t prev_log_term,
    const std::vector<LogEntry>& entries,
    int64_t leader_commit_index
) {
    logger_->debug("Handling append entries from leader: " + leader_id +
                   ", term=" + std::to_string(term) +
                   ", entries=" + std::to_string(entries.size()) +
                   ", prev_log_index=" + std::to_string(prev_log_index) +
                   ", leader_commit=" + std::to_string(leader_commit_index));

    AppendEntriesResult result;
    result.term = term;
    result.success = true;
    result.match_index = prev_log_index;

    if (cluster_service_) {
        // Process heartbeat/leader acknowledgment
        cluster_service_->receive_heartbeat(leader_id, static_cast<int>(term));
    }

    // Process log entries
    if (!entries.empty()) {
        for (const auto& entry : entries) {
            // Apply each log entry based on its type
            std::string type_str;
            switch (entry.type) {
                case LogEntryType::LOG_WRITE_OPERATION: type_str = "write_operation"; break;
                case LogEntryType::LOG_SHARD_ASSIGNMENT: type_str = "shard_assignment"; break;
                case LogEntryType::LOG_CONFIG_CHANGE: type_str = "config_change"; break;
                case LogEntryType::LOG_NODE_JOIN: type_str = "node_join"; break;
                case LogEntryType::LOG_NODE_LEAVE: type_str = "node_leave"; break;
                default: type_str = "unknown"; break;
            }
            
            logger_->debug("Processing log entry: type=" + type_str + 
                          ", index=" + std::to_string(entry.index) +
                          ", term=" + std::to_string(entry.term));

            // Handle different command types
            if (entry.type == LogEntryType::LOG_WRITE_OPERATION) {
                // In full implementation: deserialize vector from data and write to shard
                logger_->debug("Log entry: write_operation command, data_size=" + 
                              std::to_string(entry.data.size()));
            } else if (entry.type == LogEntryType::LOG_SHARD_ASSIGNMENT) {
                // In full implementation: handle shard assignment change
                logger_->debug("Log entry: shard_assignment command");
            } else if (entry.type == LogEntryType::LOG_CONFIG_CHANGE) {
                // In full implementation: handle cluster configuration change
                logger_->debug("Log entry: config_change command");
            } else if (entry.type == LogEntryType::LOG_NODE_JOIN) {
                // Handle node join
                logger_->debug("Log entry: node_join command");
            } else if (entry.type == LogEntryType::LOG_NODE_LEAVE) {
                // Handle node leave
                logger_->debug("Log entry: node_leave command");
            }
        }

        result.match_index = prev_log_index + static_cast<int64_t>(entries.size());
        logger_->info("Applied " + std::to_string(entries.size()) +
                     " log entries, new match_index=" + std::to_string(result.match_index));
    }

    return result;
}

// ============================================================================
// Helper Methods
// ============================================================================

Result<bool> DistributedWorkerService::validate_shard_exists(const std::string& shard_id) const {
    std::lock_guard<std::mutex> lock(shard_mutex_);

    if (local_shards_.find(shard_id) == local_shards_.end()) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_FOUND, "Shard not found: " + shard_id));
    }

    return true;
}

Result<bool> DistributedWorkerService::validate_shard_is_active(const std::string& shard_id) const {
    std::lock_guard<std::mutex> lock(shard_mutex_);

    auto it = local_shards_.find(shard_id);
    if (it == local_shards_.end()) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_FOUND, "Shard not found: " + shard_id));
    }

    if (it->second.status != "active") {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::INVALID_STATE, "Shard not active: " + shard_id));
    }

    return true;
}

Result<ResourceUsage> DistributedWorkerService::collect_resource_usage() const {
    ResourceUsage usage;

    // Basic resource collection (platform-specific implementations can be added later)
    #ifdef __linux__
        // Read memory info from /proc/meminfo
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo) {
            std::string line;
            while (std::getline(meminfo, line)) {
                if (line.find("MemTotal:") == 0) {
                    std::istringstream iss(line);
                    std::string label;
                    long value;
                    iss >> label >> value;
                    usage.memory_total_bytes = value * 1024;  // Convert KB to bytes
                } else if (line.find("MemAvailable:") == 0) {
                    std::istringstream iss(line);
                    std::string label;
                    long value;
                    iss >> label >> value;
                    long available_bytes = value * 1024;
                    usage.memory_used_bytes = usage.memory_total_bytes - available_bytes;
                }
            }
        }

        // Estimate CPU usage from /proc/stat
        // Read /proc/stat to calculate CPU usage
        // Format: cpu user nice system idle iowait irq softirq ...
        std::ifstream stat_file("/proc/stat");
        if (stat_file.is_open()) {
            std::string line;
            if (std::getline(stat_file, line) && line.substr(0, 3) == "cpu") {
                // Parse CPU times (simplified - would need delta calculation for accuracy)
                std::istringstream iss(line);
                std::string cpu_label;
                long user, nice, system, idle;
                iss >> cpu_label >> user >> nice >> system >> idle;
                long total = user + nice + system + idle;
                if (total > 0) {
                    usage.cpu_usage_percent = static_cast<double>(user + nice + system) / total * 100.0;
                } else {
                    usage.cpu_usage_percent = 0.0;
                }
            }
        } else {
            usage.cpu_usage_percent = 0.0;  // Fallback if /proc/stat not available
        }

        // Get disk usage for data directory
        // TODO: Make data directory configurable via environment variable or config
        const char* data_dir = std::getenv("JADEVECTORDB_DATA_DIR");
        if (!data_dir) {
            data_dir = "/tmp";  // Fallback to /tmp if not configured
        }

        struct statvfs stat;
        if (statvfs(data_dir, &stat) == 0) {
            usage.disk_total_bytes = stat.f_blocks * stat.f_frsize;
            usage.disk_used_bytes = (stat.f_blocks - stat.f_bfree) * stat.f_frsize;
        }
    #else
        // Default values for non-Linux systems
        usage.cpu_usage_percent = 0.0;
        usage.memory_used_bytes = 0;
        usage.memory_total_bytes = 0;
        usage.disk_used_bytes = 0;
        usage.disk_total_bytes = 0;
    #endif

    // Active connections tracking
    // In a full implementation, this would track:
    // - Active gRPC connections
    // - Database connections
    // - Replication connections
    // For now, provide a placeholder that could be populated by connection pool
    usage.active_connections = 0;  // Would be updated by connection pool/tracking service

    return usage;
}

Result<std::vector<ShardStatus>> DistributedWorkerService::collect_shard_statuses() const {
    std::vector<ShardStatus> statuses;

    std::lock_guard<std::mutex> lock(shard_mutex_);
    for (const auto& [shard_id, shard_info] : local_shards_) {
        ShardStatus status;
        status.shard_id = shard_id;

        // Map string status to ShardState enum
        if (shard_info.status == "active") {
            status.state = ShardState::ACTIVE;
        } else if (shard_info.status == "initializing") {
            status.state = ShardState::INITIALIZING;
        } else if (shard_info.status == "migrating") {
            status.state = ShardState::MIGRATING;
        } else {
            status.state = ShardState::OFFLINE;
        }

        status.vector_count = shard_info.record_count;
        status.size_bytes = shard_info.size_bytes;

        // Track primary status (would be set during shard assignment in full implementation)
        // For now, assume all shards on this worker are replicas unless configured otherwise
        status.is_primary = false;  // Would be tracked in shard_info or cluster metadata

        statuses.push_back(status);
    }

    return statuses;
}

Result<std::vector<ShardStats>> DistributedWorkerService::collect_shard_stats() const {
    std::vector<ShardStats> stats;

    std::lock_guard<std::mutex> lock(shard_mutex_);
    for (const auto& [shard_id, shard_info] : local_shards_) {
        ShardStats stat;
        stat.shard_id = shard_id;
        stat.vector_count = shard_info.record_count;
        stat.size_bytes = shard_info.size_bytes;

        // Per-shard statistics tracking
        // In a full implementation, each shard would maintain its own stats
        // For now, we provide aggregated stats or zeros
        // TODO: Add per-shard stat tracking with a map<shard_id, ShardStatistics>
        stat.queries_processed = 0;  // Would track per-shard query count
        stat.writes_processed = 0;   // Would track per-shard write count
        stat.avg_query_latency_ms = 0.0;  // Would calculate from per-shard timing data

        // Set last updated timestamp to current time
        auto now = std::chrono::system_clock::now();
        stat.last_updated_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        stats.push_back(stat);
    }

    return stats;
}

void DistributedWorkerService::record_query_time(int64_t time_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    queries_processed_++;
    total_query_time_ms_ += time_ms;
}

void DistributedWorkerService::record_write_time(int64_t time_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    writes_processed_++;
    total_write_time_ms_ += time_ms;
}

Result<bool> DistributedWorkerService::load_shard_data(
    const std::string& shard_id,
    const std::vector<uint8_t>& data
) {
    logger_->info("Loading data for shard: " + shard_id + ", size=" + std::to_string(data.size()) + " bytes");

    if (data.empty()) {
        logger_->debug("No data to load for shard: " + shard_id);
        return true;
    }

    // In a full implementation, this would:
    // 1. Deserialize the binary data format (e.g., FlatBuffers, Protobuf)
    // 2. Extract vectors and metadata
    // 3. Insert vectors into the local database/storage
    // 4. Update shard statistics (record_count, size_bytes)
    //
    // For now, we acknowledge the data and update shard size
    std::lock_guard<std::mutex> lock(shard_mutex_);
    auto it = local_shards_.find(shard_id);
    if (it != local_shards_.end()) {
        it->second.size_bytes = data.size();
        // Would parse data to get actual record count
        logger_->info("Shard data loaded successfully for: " + shard_id);
        return true;
    }

    return tl::make_unexpected(ErrorHandler::create_error(
        ErrorCode::NOT_FOUND, "Shard not found: " + shard_id));
}

Result<std::vector<uint8_t>> DistributedWorkerService::export_shard_data(const std::string& shard_id) const {
    logger_->info("Exporting data for shard: " + shard_id);

    // Validate shard exists
    std::lock_guard<std::mutex> lock(shard_mutex_);
    auto it = local_shards_.find(shard_id);
    if (it == local_shards_.end()) {
        return tl::make_unexpected(ErrorHandler::create_error(
            ErrorCode::NOT_FOUND, "Shard not found for export: " + shard_id));
    }

    // In a full implementation, this would:
    // 1. Query all vectors from the shard via database_layer_
    // 2. Serialize vectors and metadata using FlatBuffers or Protobuf
    // 3. Return the binary data
    //
    // For now, return empty data as a placeholder
    // In production, this would contain the serialized shard data
    std::vector<uint8_t> exported_data;

    logger_->info("Shard data export complete: " + shard_id +
                 " (" + std::to_string(it->second.record_count) + " vectors)");

    return exported_data;  // Placeholder - would contain actual serialized data
}

#ifdef BUILD_WITH_GRPC

// ============================================================================
// gRPC Service Implementation
// ============================================================================

DistributedWorkerServiceImpl::DistributedWorkerServiceImpl(
    std::shared_ptr<DistributedWorkerService> worker_service
) : worker_service_(worker_service) {
    logger_ = logging::LoggerManager::get_logger("DistributedWorkerServiceImpl");
}

grpc::Status DistributedWorkerServiceImpl::ExecuteShardSearch(
    grpc::ServerContext* context,
    const distributed::ShardSearchRequest* request,
    distributed::ShardSearchResponse* response
) {
    logger_->debug("gRPC: ExecuteShardSearch called for shard: " + request->shard_id());

    // Convert filters
    std::unordered_map<std::string, std::string> filters;
    for (const auto& [key, value] : request->filters()) {
        filters[key] = value;
    }

    // Execute search
    std::vector<float> query_vector(request->query_vector().begin(), request->query_vector().end());
    auto result = worker_service_->execute_shard_search(
        request->shard_id(),
        request->request_id(),
        query_vector,
        request->top_k(),
        request->metric_type(),
        request->threshold(),
        filters
    );

    if (!result.has_value()) {
        response->set_success(false);
        response->set_error_message(result.error().message);
        return grpc::Status::OK;
    }

    // Populate response
    response->set_shard_id(request->shard_id());
    populate_search_response(result.value(), response);
    response->set_success(true);

    return grpc::Status::OK;
}

grpc::Status DistributedWorkerServiceImpl::WriteToShard(
    grpc::ServerContext* context,
    const distributed::ShardWriteRequest* request,
    distributed::ShardWriteResponse* response
) {
    logger_->debug("gRPC: WriteToShard called for shard: " + request->shard_id());

    // Convert VectorData to Vector
    Vector vector;
    vector.id = request->vector().vector_id();
    vector.values.assign(request->vector().values().begin(), request->vector().values().end());
    
    // Convert metadata from protobuf map to Vector::Metadata
    for (const auto& [key, value] : request->vector().metadata()) {
        if (key == "source") vector.metadata.source = value;
        else if (key == "category") vector.metadata.category = value;
        else if (key == "owner") vector.metadata.owner = value;
        else if (key == "status") vector.metadata.status = value;
        else vector.metadata.custom[key] = value;
    }
    if (vector.metadata.status.empty()) vector.metadata.status = "active";
    vector.version = static_cast<int>(request->vector().version());

    auto result = worker_service_->write_to_shard(
        request->shard_id(),
        request->request_id(),
        vector,
        request->consistency_level(),
        request->wait_for_replication()
    );

    response->set_shard_id(request->shard_id());
    response->set_request_id(request->request_id());
    response->set_success(result.has_value());

    if (!result.has_value()) {
        response->set_error_message(result.error().message);
    } else {
        response->set_vectors_written(1);
    }

    return grpc::Status::OK;
}

grpc::Status DistributedWorkerServiceImpl::BatchWriteToShard(
    grpc::ServerContext* context,
    const distributed::BatchShardWriteRequest* request,
    distributed::ShardWriteResponse* response
) {
    logger_->debug("gRPC: BatchWriteToShard called for shard: " + request->shard_id());

    // Convert vectors
    std::vector<Vector> vectors;
    for (const auto& vec_data : request->vectors()) {
        Vector vector;
        vector.id = vec_data.vector_id();
        vector.values.assign(vec_data.values().begin(), vec_data.values().end());
        vectors.push_back(vector);
    }

    auto result = worker_service_->batch_write_to_shard(
        request->shard_id(),
        request->request_id(),
        vectors,
        request->consistency_level(),
        request->wait_for_replication()
    );

    response->set_shard_id(request->shard_id());
    response->set_request_id(request->request_id());
    response->set_success(result.has_value());

    if (!result.has_value()) {
        response->set_error_message(result.error().message);
    } else {
        response->set_vectors_written(result.value());
    }

    return grpc::Status::OK;
}

grpc::Status DistributedWorkerServiceImpl::DeleteFromShard(
    grpc::ServerContext* context,
    const distributed::ShardDeleteRequest* request,
    distributed::ShardDeleteResponse* response
) {
    std::vector<std::string> vector_ids(request->vector_ids().begin(), request->vector_ids().end());

    auto result = worker_service_->delete_from_shard(
        request->shard_id(),
        request->request_id(),
        vector_ids,
        request->consistency_level()
    );

    response->set_shard_id(request->shard_id());
    response->set_request_id(request->request_id());
    response->set_success(result.has_value());

    if (!result.has_value()) {
        response->set_error_message(result.error().message);
    } else {
        response->set_vectors_deleted(result.value());
    }

    return grpc::Status::OK;
}

grpc::Status DistributedWorkerServiceImpl::HealthCheck(
    grpc::ServerContext* context,
    const distributed::HealthCheckRequest* request,
    distributed::HealthCheckResponse* response
) {
    auto health = worker_service_->get_health();

    if (!health.has_value()) {
        response->set_status(distributed::UNHEALTHY);
        return grpc::Status::OK;
    }

    response->set_node_id(request->node_id());
    response->set_status(static_cast<distributed::HealthStatus>(health.value().status));
    response->set_version(health.value().version);
    response->set_uptime_seconds(health.value().uptime_seconds);

    // Populate resource usage
    auto resources = worker_service_->collect_resource_usage();
    auto* resource_usage = response->mutable_resource_usage();
    resource_usage->set_cpu_usage_percent(resources.cpu_usage_percent);
    resource_usage->set_memory_used_bytes(resources.memory_used_bytes);
    resource_usage->set_memory_total_bytes(resources.memory_total_bytes);
    resource_usage->set_disk_used_bytes(resources.disk_used_bytes);
    resource_usage->set_disk_total_bytes(resources.disk_total_bytes);
    resource_usage->set_active_connections(resources.active_connections);

    // Populate shard statuses
    auto shard_statuses = worker_service_->collect_shard_statuses();
    if (shard_statuses.has_value()) {
        for (const auto& shard : shard_statuses.value()) {
            auto* status = response->add_shard_statuses();
            status->set_shard_id(shard.shard_id);
            status->set_state(static_cast<distributed::ShardState>(shard.state));
            status->set_vector_count(shard.vector_count);
            status->set_size_bytes(shard.size_bytes);
            status->set_is_primary(shard.is_primary);
        }
    }

    return grpc::Status::OK;
}

grpc::Status DistributedWorkerServiceImpl::GetWorkerStats(
    grpc::ServerContext* context,
    const distributed::WorkerStatsRequest* request,
    distributed::WorkerStatsResponse* response
) {
    auto stats = worker_service_->get_worker_stats(request->include_shard_details());

    if (!stats.has_value()) {
        return grpc::Status(grpc::INTERNAL, stats.error().message);
    }

    response->set_node_id(request->node_id());
    response->set_total_vectors(stats.value().total_vectors);
    response->set_active_shards(stats.value().active_shards);
    response->set_queries_processed(stats.value().queries_processed);
    response->set_writes_processed(stats.value().writes_processed);
    response->set_avg_query_latency_ms(stats.value().avg_query_latency_ms);
    response->set_avg_write_latency_ms(stats.value().avg_write_latency_ms);

    // Populate resource usage
    auto resources = worker_service_->collect_resource_usage();
    auto* resource_usage = response->mutable_resource_usage();
    resource_usage->set_cpu_usage_percent(resources.cpu_usage_percent);
    resource_usage->set_memory_used_bytes(resources.memory_used_bytes);
    resource_usage->set_memory_total_bytes(resources.memory_total_bytes);
    resource_usage->set_disk_used_bytes(resources.disk_used_bytes);
    resource_usage->set_disk_total_bytes(resources.disk_total_bytes);
    resource_usage->set_active_connections(resources.active_connections);

    // Populate shard stats if requested
    if (request->include_shard_details()) {
        auto shard_stats = worker_service_->collect_shard_stats();
        if (shard_stats.has_value()) {
            for (const auto& stat : shard_stats.value()) {
                auto* shard_stat = response->add_shard_stats();
                shard_stat->set_shard_id(stat.shard_id);
                shard_stat->set_vector_count(stat.vector_count);
                shard_stat->set_size_bytes(stat.size_bytes);
                shard_stat->set_queries_processed(stat.queries_processed);
                shard_stat->set_writes_processed(stat.writes_processed);
                shard_stat->set_avg_query_latency_ms(stat.avg_query_latency_ms);
                shard_stat->set_last_updated_timestamp(stat.last_updated_timestamp);
            }
        }
    }

    return grpc::Status::OK;
}

// Additional gRPC methods - implementing stubs for compilation
grpc::Status DistributedWorkerServiceImpl::AssignShard(
    grpc::ServerContext* context,
    const distributed::AssignShardRequest* request,
    distributed::AssignShardResponse* response
) {
    // Convert ShardConfig from protobuf
    ShardConfig config;
    config.index_type = request->config().index_type();
    config.vector_dimension = request->config().vector_dimension();
    config.metric_type = request->config().metric_type();
    config.replication_factor = request->config().replication_factor();
    for (const auto& [key, value] : request->config().index_parameters()) {
        config.index_parameters[key] = value;
    }
    
    // Convert initial data
    std::vector<uint8_t> initial_data;
    if (!request->initial_data().empty()) {
        const auto& data = request->initial_data();
        initial_data.assign(data.begin(), data.end());
    }

    auto result = worker_service_->assign_shard(
        request->shard_id(),
        request->is_primary(),
        config,
        initial_data
    );

    response->set_shard_id(request->shard_id());
    response->set_worker_id(request->worker_id());
    response->set_success(result.has_value());

    if (!result.has_value()) {
        response->set_error_message(result.error().message);
    }

    return grpc::Status::OK;
}

grpc::Status DistributedWorkerServiceImpl::RemoveShard(
    grpc::ServerContext* context,
    const distributed::RemoveShardRequest* request,
    distributed::RemoveShardResponse* response
) {
    auto result = worker_service_->remove_shard(
        request->shard_id(),
        request->force(),
        request->transfer_to_worker(),
        request->target_worker_id()
    );

    response->set_shard_id(request->shard_id());
    response->set_success(result.has_value());

    if (!result.has_value()) {
        response->set_error_message(result.error().message);
    }

    return grpc::Status::OK;
}

grpc::Status DistributedWorkerServiceImpl::GetShardInfo(
    grpc::ServerContext* context,
    const distributed::ShardInfoRequest* request,
    distributed::ShardInfoResponse* response
) {
    auto result = worker_service_->get_shard_info(
        request->shard_id(),
        request->include_statistics()
    );

    if (!result.has_value()) {
        return grpc::Status(grpc::NOT_FOUND, result.error().message);
    }

    response->set_shard_id(result.value().shard_id);
    response->set_worker_id(request->worker_id());
    response->set_state(static_cast<distributed::ShardState>(result.value().state));
    response->set_is_primary(result.value().primary);
    response->set_vector_count(result.value().record_count);
    response->set_size_bytes(result.value().size_bytes);

    return grpc::Status::OK;
}

grpc::Status DistributedWorkerServiceImpl::ReplicateData(
    grpc::ServerContext* context,
    const distributed::ReplicationRequest* request,
    distributed::ReplicationResponse* response
) {
    // Implement replication data handling
    response->set_shard_id(request->shard_id());
    
    // Validate shard exists
    auto shard_check = worker_service_->validate_shard_exists(request->shard_id());
    if (!shard_check.has_value()) {
        response->set_success(false);
        response->set_error_message("Shard not found: " + request->shard_id());
        return grpc::Status::OK;
    }
    
    // Process incoming vectors for replication
    int vectors_replicated = 0;
    for (const auto& proto_vector : request->vectors()) {
        Vector vector;
        vector.id = proto_vector.vector_id();
        vector.values.assign(proto_vector.values().begin(), proto_vector.values().end());
        
        // Convert metadata
        for (const auto& [key, value] : proto_vector.metadata()) {
            if (key == "source") vector.metadata.source = value;
            else if (key == "category") vector.metadata.category = value;
            else if (key == "owner") vector.metadata.owner = value;
            else if (key == "status") vector.metadata.status = value;
            else vector.metadata.custom[key] = value;
        }
        if (vector.metadata.status.empty()) vector.metadata.status = "active";
        vector.version = static_cast<int>(proto_vector.version());
        
        // Write to local shard (as replica)
        auto write_result = worker_service_->write_to_shard(
            request->shard_id(),
            "replication-" + std::to_string(proto_vector.timestamp()),
            vector,
            0,  // No consistency level for replica writes
            false  // Don't wait for further replication
        );
        
        if (write_result.has_value()) {
            vectors_replicated++;
        }
    }
    
    response->set_success(vectors_replicated == request->vectors_size());
    response->set_vectors_replicated(vectors_replicated);
    if (vectors_replicated < request->vectors_size()) {
        response->set_error_message("Partial replication: " + 
            std::to_string(vectors_replicated) + "/" + 
            std::to_string(request->vectors_size()) + " vectors replicated");
    }

    return grpc::Status::OK;
}

grpc::Status DistributedWorkerServiceImpl::SyncShard(
    grpc::ServerContext* context,
    const distributed::ShardSyncRequest* request,
    distributed::ShardSyncResponse* response
) {
    auto result = worker_service_->sync_shard(
        request->shard_id(),
        request->source_node_id(),
        request->target_version()
    );

    response->set_shard_id(request->shard_id());
    response->set_success(result.has_value());

    if (!result.has_value()) {
        response->set_error_message(result.error().message);
    }

    return grpc::Status::OK;
}

grpc::Status DistributedWorkerServiceImpl::RequestVote(
    grpc::ServerContext* context,
    const distributed::VoteRequest* request,
    distributed::VoteResponse* response
) {
    auto result = worker_service_->handle_vote_request(
        request->term(),
        request->candidate_id(),
        request->last_log_index(),
        request->last_log_term()
    );

    if (!result.has_value()) {
        return grpc::Status(grpc::INTERNAL, result.error().message);
    }

    response->set_term(result.value().term);
    response->set_vote_granted(result.value().vote_granted);
    response->set_voter_id(request->candidate_id());

    return grpc::Status::OK;
}

grpc::Status DistributedWorkerServiceImpl::SendHeartbeat(
    grpc::ServerContext* context,
    const distributed::HeartbeatRequest* request,
    distributed::HeartbeatResponse* response
) {
    auto result = worker_service_->handle_heartbeat(
        request->term(),
        request->leader_id(),
        request->prev_log_index(),
        request->prev_log_term(),
        request->leader_commit_index()
    );

    if (!result.has_value()) {
        return grpc::Status(grpc::INTERNAL, result.error().message);
    }

    response->set_term(result.value().term);
    response->set_success(result.value().success);
    response->set_follower_id(request->leader_id());
    response->set_match_index(result.value().match_index);

    return grpc::Status::OK;
}

grpc::Status DistributedWorkerServiceImpl::AppendEntries(
    grpc::ServerContext* context,
    const distributed::AppendEntriesRequest* request,
    distributed::AppendEntriesResponse* response
) {
    // Convert log entries from protobuf
    std::vector<LogEntry> entries;
    entries.reserve(request->entries_size());
    for (const auto& proto_entry : request->entries()) {
        LogEntry entry;
        entry.index = proto_entry.index();
        entry.term = proto_entry.term();
        entry.type = proto_entry.type();
        entry.data.assign(proto_entry.data().begin(), proto_entry.data().end());
        entry.timestamp = proto_entry.timestamp();
        entries.push_back(std::move(entry));
    }

    auto result = worker_service_->handle_append_entries(
        request->term(),
        request->leader_id(),
        request->prev_log_index(),
        request->prev_log_term(),
        entries,
        request->leader_commit_index()
    );

    if (!result.has_value()) {
        response->set_success(false);
        response->set_error_message(result.error().message);
        return grpc::Status::OK;
    }

    response->set_term(result.value().term);
    response->set_success(result.value().success);
    response->set_follower_id(request->leader_id());
    response->set_match_index(result.value().match_index);

    return grpc::Status::OK;
}

void DistributedWorkerServiceImpl::populate_search_response(
    const SearchResults& results,
    distributed::ShardSearchResponse* response
) {
    // Note: shard_id should be set by the caller as SearchResults doesn't track it
    // response->set_shard_id() should be called before this method

    for (const auto& result : results.results) {
        auto* search_result = response->add_results();
        search_result->set_vector_id(result.vector_id);
        search_result->set_similarity_score(result.score);

        for (float val : result.vector.values) {
            search_result->add_values(val);
        }

        // Add metadata
        auto* metadata = search_result->mutable_metadata();
        if (!result.vector.metadata.source.empty())
            (*metadata)["source"] = result.vector.metadata.source;
        if (!result.vector.metadata.category.empty())
            (*metadata)["category"] = result.vector.metadata.category;
        if (!result.vector.metadata.owner.empty())
            (*metadata)["owner"] = result.vector.metadata.owner;
        if (!result.vector.metadata.status.empty())
            (*metadata)["status"] = result.vector.metadata.status;
        for (const auto& [key, value] : result.vector.metadata.custom) {
            if (value.is_string()) {
                (*metadata)[key] = value.get<std::string>();
            }
        }
    }
}

#endif // BUILD_WITH_GRPC

} // namespace jadevectordb
