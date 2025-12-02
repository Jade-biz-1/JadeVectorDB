#include "distributed_worker_service.h"
#include "models/vector_database.h"
#include <chrono>
#include <algorithm>
#include <sstream>

namespace jadevectordb {

DistributedWorkerService::DistributedWorkerService(
    const std::string& node_id,
    const std::string& host,
    int port,
    std::shared_ptr<VectorDatabase> vector_db,
    std::shared_ptr<SimilaritySearchService> search_service,
    std::shared_ptr<ShardingService> sharding_service,
    std::shared_ptr<ClusterService> cluster_service
) : node_id_(node_id),
    node_host_(host),
    node_port_(port),
    vector_db_(vector_db),
    search_service_(search_service),
    sharding_service_(sharding_service),
    cluster_service_(cluster_service),
    queries_processed_(0),
    writes_processed_(0),
    total_query_time_ms_(0),
    total_write_time_ms_(0),
    initialized_(false),
    running_(false) {
    logger_ = logging::get_logger("DistributedWorkerService");
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
        return create_error(ErrorCode::INVALID_STATE, "Worker service already initialized");
    }

    logger_->info("Initializing distributed worker service for node: " + node_id_);

    // Validate dependencies
    if (!vector_db_) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "VectorDatabase not provided");
    }
    if (!search_service_) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "SimilaritySearchService not provided");
    }
    if (!sharding_service_) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "ShardingService not provided");
    }
    if (!cluster_service_) {
        return create_error(ErrorCode::INVALID_ARGUMENT, "ClusterService not provided");
    }

    initialized_ = true;
    logger_->info("Distributed worker service initialized successfully");
    return true;
}

Result<bool> DistributedWorkerService::start() {
    if (!initialized_) {
        return create_error(ErrorCode::INVALID_STATE, "Worker service not initialized");
    }

    if (running_) {
        return create_error(ErrorCode::INVALID_STATE, "Worker service already running");
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
        return create_error(ErrorCode::INTERNAL_ERROR, "Failed to start gRPC server");
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
        return create_error(ErrorCode::INVALID_STATE, "Worker service not running");
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
    SearchRequest search_req;
    search_req.query_vector = query_vector;
    search_req.top_k = top_k;
    search_req.metric_type = metric_type;
    search_req.threshold = threshold;

    // Get database ID from shard
    std::string database_id;
    {
        std::lock_guard<std::mutex> lock(shard_mutex_);
        auto it = local_shards_.find(shard_id);
        if (it != local_shards_.end()) {
            database_id = it->second.database_id;
        }
    }

    auto search_result = search_service_->search(database_id, search_req);

    auto end = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    record_query_time(duration_ms);

    if (!search_result.has_value()) {
        logger_->error("Shard search failed: " + search_result.error().message);
        return tl::unexpected(search_result.error());
    }

    logger_->debug("Shard search completed: " + std::to_string(search_result.value().results.size()) +
                   " results in " + std::to_string(duration_ms) + "ms");

    return search_result.value();
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
    auto write_result = vector_db_->store_vector(database_id, vector);

    auto end = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    record_write_time(duration_ms);

    if (!write_result.has_value()) {
        logger_->error("Shard write failed: " + write_result.error().message);
        return tl::unexpected(write_result.error());
    }

    // Handle replication if needed
    if (wait_for_replication && consistency_level != ConsistencyLevel::EVENTUAL) {
        // TODO: Implement synchronous replication waiting
        logger_->debug("Waiting for replication (not yet implemented)");
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
        auto result = vector_db_->store_vector(database_id, vector);
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
        auto result = vector_db_->delete_vector(database_id, vector_id);
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
    health.version = "1.0.0";  // TODO: Get from build system

    auto now = std::chrono::steady_clock::now();
    health.uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();

    auto resource_usage = collect_resource_usage();
    if (resource_usage.has_value()) {
        health.resource_usage = resource_usage.value();
    }

    auto shard_statuses = collect_shard_statuses();
    if (shard_statuses.has_value()) {
        health.shard_statuses = shard_statuses.value();
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
            stats.total_vectors += shard_info.vector_count;
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
        return create_error(ErrorCode::ALREADY_EXISTS, "Shard already assigned to this worker");
    }

    ShardInfo shard_info;
    shard_info.shard_id = shard_id;
    shard_info.is_primary = is_primary;
    shard_info.state = ShardState::INITIALIZING;
    shard_info.vector_count = 0;
    shard_info.size_bytes = 0;
    shard_info.database_id = config.database_id;  // Assuming config has database_id

    local_shards_[shard_id] = shard_info;

    // Load initial data if provided
    if (!initial_data.empty()) {
        auto load_result = load_shard_data(shard_id, initial_data);
        if (!load_result.has_value()) {
            local_shards_.erase(shard_id);
            return tl::unexpected(load_result.error());
        }
    }

    local_shards_[shard_id].state = ShardState::ACTIVE;

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
        return create_error(ErrorCode::NOT_FOUND, "Shard not found on this worker");
    }

    // If transfer requested, export data first
    if (transfer_to_worker && !target_worker_id.empty()) {
        logger_->info("Transferring shard " + shard_id + " to worker " + target_worker_id);
        // TODO: Implement data transfer
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
        return create_error(ErrorCode::NOT_FOUND, "Shard not found on this worker");
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
    result.replication_lag_ms = 0;  // TODO: Calculate actual lag
    result.current_version = to_version;

    logger_->info("Replication completed: " + std::to_string(result.vectors_replicated) + " vectors");
    return result;
}

Result<bool> DistributedWorkerService::sync_shard(
    const std::string& shard_id,
    const std::string& source_node_id,
    int64_t target_version
) {
    logger_->info("Syncing shard: " + shard_id + " with node " + source_node_id);

    // TODO: Implement shard synchronization
    logger_->warn("Shard sync not yet implemented");

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
    logger_->debug("Handling vote request from: " + candidate_id + ", term=" + std::to_string(term));

    // Delegate to cluster service
    auto vote_result = cluster_service_->handle_vote_request(term, candidate_id, last_log_index, last_log_term);

    if (!vote_result.has_value()) {
        return tl::unexpected(vote_result.error());
    }

    VoteResult result;
    result.term = term;
    result.vote_granted = vote_result.value();

    logger_->debug("Vote result: " + std::string(result.vote_granted ? "granted" : "denied"));
    return result;
}

Result<DistributedWorkerService::HeartbeatResult> DistributedWorkerService::handle_heartbeat(
    int64_t term,
    const std::string& leader_id,
    int64_t prev_log_index,
    int64_t prev_log_term,
    int64_t leader_commit_index
) {
    logger_->debug("Handling heartbeat from leader: " + leader_id);

    // Delegate to cluster service
    auto hb_result = cluster_service_->handle_heartbeat(term, leader_id, prev_log_index,
                                                        prev_log_term, leader_commit_index);

    if (!hb_result.has_value()) {
        return tl::unexpected(hb_result.error());
    }

    HeartbeatResult result;
    result.term = term;
    result.success = hb_result.value();
    result.match_index = prev_log_index;

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
                   ", entries=" + std::to_string(entries.size()));

    // Delegate to cluster service
    auto append_result = cluster_service_->handle_append_entries(term, leader_id, prev_log_index,
                                                                 prev_log_term, entries, leader_commit_index);

    if (!append_result.has_value()) {
        return tl::unexpected(append_result.error());
    }

    AppendEntriesResult result;
    result.term = term;
    result.success = append_result.value();
    result.match_index = prev_log_index + entries.size();

    return result;
}

// ============================================================================
// Helper Methods
// ============================================================================

Result<bool> DistributedWorkerService::validate_shard_exists(const std::string& shard_id) const {
    std::lock_guard<std::mutex> lock(shard_mutex_);

    if (local_shards_.find(shard_id) == local_shards_.end()) {
        return create_error(ErrorCode::NOT_FOUND, "Shard not found: " + shard_id);
    }

    return true;
}

Result<bool> DistributedWorkerService::validate_shard_is_active(const std::string& shard_id) const {
    std::lock_guard<std::mutex> lock(shard_mutex_);

    auto it = local_shards_.find(shard_id);
    if (it == local_shards_.end()) {
        return create_error(ErrorCode::NOT_FOUND, "Shard not found: " + shard_id);
    }

    if (it->second.state != ShardState::ACTIVE) {
        return create_error(ErrorCode::INVALID_STATE, "Shard not active: " + shard_id);
    }

    return true;
}

Result<ResourceUsage> DistributedWorkerService::collect_resource_usage() const {
    ResourceUsage usage;
    // TODO: Implement actual resource collection
    usage.cpu_usage_percent = 0.0;
    usage.memory_used_bytes = 0;
    usage.memory_total_bytes = 0;
    usage.disk_used_bytes = 0;
    usage.disk_total_bytes = 0;
    usage.active_connections = 0;
    return usage;
}

Result<std::vector<ShardStatus>> DistributedWorkerService::collect_shard_statuses() const {
    std::vector<ShardStatus> statuses;

    std::lock_guard<std::mutex> lock(shard_mutex_);
    for (const auto& [shard_id, shard_info] : local_shards_) {
        ShardStatus status;
        status.shard_id = shard_id;
        status.state = shard_info.state;
        status.vector_count = shard_info.vector_count;
        status.size_bytes = shard_info.size_bytes;
        status.is_primary = shard_info.is_primary;
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
        stat.vector_count = shard_info.vector_count;
        stat.size_bytes = shard_info.size_bytes;
        stat.queries_processed = 0;  // TODO: Track per-shard stats
        stat.writes_processed = 0;
        stat.avg_query_latency_ms = 0.0;
        stat.last_updated_timestamp = 0;
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
    // TODO: Implement data loading from binary format
    return true;
}

Result<std::vector<uint8_t>> DistributedWorkerService::export_shard_data(const std::string& shard_id) const {
    logger_->info("Exporting data for shard: " + shard_id);
    // TODO: Implement data export to binary format
    return std::vector<uint8_t>();
}

#ifdef BUILD_WITH_GRPC

// ============================================================================
// gRPC Service Implementation
// ============================================================================

DistributedWorkerServiceImpl::DistributedWorkerServiceImpl(
    std::shared_ptr<DistributedWorkerService> worker_service
) : worker_service_(worker_service) {
    logger_ = logging::get_logger("DistributedWorkerServiceImpl");
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
    // TODO: Convert metadata

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

    // TODO: Populate resource usage and shard statuses

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

    // TODO: Populate resource usage and shard stats

    return grpc::Status::OK;
}

// Additional gRPC methods - implementing stubs for compilation
grpc::Status DistributedWorkerServiceImpl::AssignShard(
    grpc::ServerContext* context,
    const distributed::AssignShardRequest* request,
    distributed::AssignShardResponse* response
) {
    // TODO: Implement full conversion
    ShardConfig config;
    std::vector<uint8_t> initial_data;

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
    response->set_is_primary(result.value().is_primary);
    response->set_vector_count(result.value().vector_count);
    response->set_size_bytes(result.value().size_bytes);

    return grpc::Status::OK;
}

grpc::Status DistributedWorkerServiceImpl::ReplicateData(
    grpc::ServerContext* context,
    const distributed::ReplicationRequest* request,
    distributed::ReplicationResponse* response
) {
    // TODO: Full implementation
    response->set_shard_id(request->shard_id());
    response->set_success(false);
    response->set_error_message("Not yet implemented");

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
    // TODO: Convert log entries
    std::vector<LogEntry> entries;

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
    response->set_shard_id(results.database_id);  // Using database_id as placeholder

    for (const auto& result : results.results) {
        auto* search_result = response->add_results();
        search_result->set_vector_id(result.vector_id);
        search_result->set_similarity_score(result.score);

        for (float val : result.vector.values) {
            search_result->add_values(val);
        }

        // TODO: Add metadata
    }
}

#endif // BUILD_WITH_GRPC

} // namespace jadevectordb
