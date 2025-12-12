#include "distributed_master_client.h"
#include <thread>
#include <algorithm>
#include <future>

namespace jadevectordb {

DistributedMasterClient::DistributedMasterClient()
    : config_(),
      total_requests_(0),
      failed_requests_(0),
      total_request_time_ms_(0),
      initialized_(false) {
    logger_ = logging::LoggerManager::get_logger("DistributedMasterClient");
}

DistributedMasterClient::DistributedMasterClient(const RpcConfig& config)
    : config_(config),
      total_requests_(0),
      failed_requests_(0),
      total_request_time_ms_(0),
      initialized_(false) {
    logger_ = logging::LoggerManager::get_logger("DistributedMasterClient");
}

DistributedMasterClient::~DistributedMasterClient() {
    if (initialized_) {
        shutdown();
    }
}

// ============================================================================
// Lifecycle Methods
// ============================================================================

Result<bool> DistributedMasterClient::initialize() {
    if (initialized_) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::INVALID_STATE, "Client already initialized"));
    }

    logger_->info("Initializing distributed master client");

    initialized_ = true;
    logger_->info("Distributed master client initialized");
    return true;
}

Result<bool> DistributedMasterClient::shutdown() {
    if (!initialized_) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::INVALID_STATE, "Client not initialized"));
    }

    logger_->info("Shutting down distributed master client");

    // Clear all connections
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        worker_connections_.clear();
    }

    initialized_ = false;
    logger_->info("Distributed master client shut down");
    return true;
}

// ============================================================================
// Connection Management
// ============================================================================

Result<bool> DistributedMasterClient::add_worker(
    const std::string& worker_id,
    const std::string& host,
    int port
) {
    logger_->info("Adding worker: " + worker_id + " at " + host + ":" + std::to_string(port));

    std::lock_guard<std::mutex> lock(connections_mutex_);

    if (worker_connections_.find(worker_id) != worker_connections_.end()) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::ALREADY_EXISTS, "Worker already exists: " + worker_id));
    }

    auto connection = std::make_shared<WorkerConnection>();
    connection->worker_id = worker_id;
    connection->host = host;
    connection->port = port;
    connection->is_active = true;
    connection->last_success = std::chrono::steady_clock::now();
    connection->consecutive_failures = 0;

#ifdef BUILD_WITH_GRPC
    connection->channel = create_channel(host, port);
    if (!connection->channel) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NETWORK_ERROR, "Failed to create gRPC channel for worker: " + worker_id));
    }

    connection->stub = distributed::DistributedService::NewStub(connection->channel);
    if (!connection->stub) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NETWORK_ERROR, "Failed to create gRPC stub for worker: " + worker_id));
    }
#endif

    worker_connections_[worker_id] = connection;

    logger_->info("Worker added successfully: " + worker_id);
    return true;
}

Result<bool> DistributedMasterClient::remove_worker(const std::string& worker_id) {
    logger_->info("Removing worker: " + worker_id);

    std::lock_guard<std::mutex> lock(connections_mutex_);

    auto it = worker_connections_.find(worker_id);
    if (it == worker_connections_.end()) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_FOUND, "Worker not found: " + worker_id));
    }

    worker_connections_.erase(it);

    logger_->info("Worker removed: " + worker_id);
    return true;
}

std::vector<std::string> DistributedMasterClient::get_connected_workers() const {
    std::lock_guard<std::mutex> lock(connections_mutex_);

    std::vector<std::string> workers;
    for (const auto& [worker_id, connection] : worker_connections_) {
        if (connection->is_active) {
            workers.push_back(worker_id);
        }
    }

    return workers;
}

bool DistributedMasterClient::is_worker_connected(const std::string& worker_id) const {
    std::lock_guard<std::mutex> lock(connections_mutex_);

    auto it = worker_connections_.find(worker_id);
    return it != worker_connections_.end() && it->second->is_active;
}

// ============================================================================
// Search Operations
// ============================================================================

Result<DistributedMasterClient::SearchResponse> DistributedMasterClient::execute_shard_search(
    const std::string& worker_id,
    const SearchRequest& request
) {
    auto start = std::chrono::steady_clock::now();

    logger_->debug("Executing shard search on worker: " + worker_id + ", shard: " + request.shard_id);

    auto connection_result = get_worker_connection(worker_id);
    if (!connection_result.has_value()) {
        record_request(false, 0);
        return tl::unexpected(connection_result.error());
    }

#ifdef BUILD_WITH_GRPC
    auto connection = connection_result.value();

    distributed::ShardSearchRequest grpc_request;
    grpc_request.set_shard_id(request.shard_id);
    grpc_request.set_request_id(request.request_id);

    for (float val : request.query_vector) {
        grpc_request.add_query_vector(val);
    }

    grpc_request.set_top_k(request.top_k);
    grpc_request.set_metric_type(request.metric_type);
    grpc_request.set_threshold(request.threshold);
    grpc_request.set_timeout_ms(request.timeout.count());

    for (const auto& [key, value] : request.filters) {
        (*grpc_request.mutable_filters())[key] = value;
    }

    distributed::ShardSearchResponse grpc_response;
    auto context = create_context(request.timeout);

    grpc::Status status = connection->stub->ExecuteShardSearch(context.get(), grpc_request, &grpc_response);

    auto end = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (!status.ok()) {
        mark_worker_failed(worker_id);
        record_request(false, duration_ms);
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::RPC_ERROR, "RPC failed: " + status.error_message()));)
    }

    mark_worker_success(worker_id);
    record_request(true, duration_ms);

    SearchResponse response;
    response.shard_id = grpc_response.shard_id();
    response.request_id = grpc_response.request_id();
    response.success = grpc_response.success();
    response.error_message = grpc_response.error_message();
    response.execution_time_ms = grpc_response.execution_time_ms();
    response.vectors_scanned = grpc_response.vectors_scanned();

    for (const auto& result : grpc_response.results()) {
        SearchResult sr;
        sr.vector_id = result.vector_id();
        sr.score = result.similarity_score();
        sr.vector.values.assign(result.values().begin(), result.values().end());
        response.results.push_back(sr);
    }

    logger_->debug("Search completed: " + std::to_string(response.results.size()) + " results");
    return response;
#else
    logger_->warn("gRPC not enabled, search operation not available");
    return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_IMPLEMENTED, "gRPC not enabled"));
#endif
}

Result<std::vector<DistributedMasterClient::SearchResponse>> DistributedMasterClient::execute_distributed_search(
    const std::vector<std::string>& worker_ids,
    const SearchRequest& request
) {
    logger_->info("Executing distributed search across " + std::to_string(worker_ids.size()) + " workers");

    std::vector<std::future<Result<SearchResponse>>> futures;

    // Launch parallel searches
    for (const auto& worker_id : worker_ids) {
        futures.push_back(std::async(std::launch::async, [this, worker_id, request]() {
            return execute_shard_search(worker_id, request);
        }));
    }

    // Collect results
    std::vector<SearchResponse> responses;
    bool all_succeeded = true;

    for (auto& future : futures) {
        try {
            auto result = future.get();
            if (result.has_value()) {
                responses.push_back(result.value());
            } else {
                logger_->warn("Worker search failed: " + result.error().message);
                all_succeeded = false;
            }
        } catch (const std::exception& e) {
            logger_->error("Exception in distributed search: " + std::string(e.what()));
            all_succeeded = false;
        }
    }

    logger_->info("Distributed search completed: " + std::to_string(responses.size()) + "/" +
                  std::to_string(worker_ids.size()) + " successful");

    if (responses.empty()) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::INTERNAL_ERROR, "All worker searches failed"));
    }

    return responses;
}

// ============================================================================
// Write Operations
// ============================================================================

Result<DistributedMasterClient::WriteResponse> DistributedMasterClient::write_to_shard(
    const std::string& worker_id,
    const WriteRequest& request
) {
    auto start = std::chrono::steady_clock::now();

    logger_->debug("Writing to shard on worker: " + worker_id + ", shard: " + request.shard_id);

    auto connection_result = get_worker_connection(worker_id);
    if (!connection_result.has_value()) {
        record_request(false, 0);
        return tl::unexpected(connection_result.error());
    }

#ifdef BUILD_WITH_GRPC
    auto connection = connection_result.value();

    distributed::ShardWriteRequest grpc_request;
    grpc_request.set_shard_id(request.shard_id);
    grpc_request.set_request_id(request.request_id);
    grpc_request.set_consistency_level(static_cast<distributed::ConsistencyLevel>(request.consistency_level));
    grpc_request.set_wait_for_replication(request.wait_for_replication);

    auto* vector_data = grpc_request.mutable_vector();
    vector_data->set_vector_id(request.vector.id);

    for (float val : request.vector.values) {
        vector_data->add_values(val);
    }

    distributed::ShardWriteResponse grpc_response;
    auto context = create_context(config_.write_timeout);

    grpc::Status status = connection->stub->WriteToShard(context.get(), grpc_request, &grpc_response);

    auto end = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (!status.ok()) {
        mark_worker_failed(worker_id);
        record_request(false, duration_ms);
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::RPC_ERROR, "RPC failed: " + status.error_message()));)
    }

    mark_worker_success(worker_id);
    record_request(true, duration_ms);

    WriteResponse response;
    response.shard_id = grpc_response.shard_id();
    response.success = grpc_response.success();
    response.error_message = grpc_response.error_message();
    response.execution_time_ms = grpc_response.execution_time_ms();

    return response;
#else
    logger_->warn("gRPC not enabled, write operation not available");
    return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_IMPLEMENTED, "gRPC not enabled"));
#endif
}

Result<DistributedMasterClient::WriteResponse> DistributedMasterClient::batch_write_to_shard(
    const std::string& worker_id,
    const std::string& shard_id,
    const std::string& request_id,
    const std::vector<Vector>& vectors,
    ConsistencyLevel consistency_level
) {
    auto start = std::chrono::steady_clock::now();

    logger_->debug("Batch writing " + std::to_string(vectors.size()) + " vectors to worker: " + worker_id);

    auto connection_result = get_worker_connection(worker_id);
    if (!connection_result.has_value()) {
        record_request(false, 0);
        return tl::unexpected(connection_result.error());
    }

#ifdef BUILD_WITH_GRPC
    auto connection = connection_result.value();

    distributed::BatchShardWriteRequest grpc_request;
    grpc_request.set_shard_id(shard_id);
    grpc_request.set_request_id(request_id);
    grpc_request.set_consistency_level(static_cast<distributed::ConsistencyLevel>(consistency_level));
    grpc_request.set_wait_for_replication(false);

    for (const auto& vector : vectors) {
        auto* vector_data = grpc_request.add_vectors();
        vector_data->set_vector_id(vector.id);

        for (float val : vector.values) {
            vector_data->add_values(val);
        }
    }

    distributed::ShardWriteResponse grpc_response;
    auto context = create_context(config_.write_timeout);

    grpc::Status status = connection->stub->BatchWriteToShard(context.get(), grpc_request, &grpc_response);

    auto end = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (!status.ok()) {
        mark_worker_failed(worker_id);
        record_request(false, duration_ms);
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::RPC_ERROR, "RPC failed: " + status.error_message()));)
    }

    mark_worker_success(worker_id);
    record_request(true, duration_ms);

    WriteResponse response;
    response.shard_id = grpc_response.shard_id();
    response.success = grpc_response.success();
    response.error_message = grpc_response.error_message();
    response.execution_time_ms = grpc_response.execution_time_ms();

    return response;
#else
    logger_->warn("gRPC not enabled, batch write operation not available");
    return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_IMPLEMENTED, "gRPC not enabled"));
#endif
}

Result<bool> DistributedMasterClient::delete_from_shard(
    const std::string& worker_id,
    const std::string& shard_id,
    const std::string& request_id,
    const std::vector<std::string>& vector_ids,
    ConsistencyLevel consistency_level
) {
    logger_->debug("Deleting " + std::to_string(vector_ids.size()) + " vectors from worker: " + worker_id);

    auto connection_result = get_worker_connection(worker_id);
    if (!connection_result.has_value()) {
        return tl::unexpected(connection_result.error());
    }

#ifdef BUILD_WITH_GRPC
    auto connection = connection_result.value();

    distributed::ShardDeleteRequest grpc_request;
    grpc_request.set_shard_id(shard_id);
    grpc_request.set_request_id(request_id);
    grpc_request.set_consistency_level(static_cast<distributed::ConsistencyLevel>(consistency_level));

    for (const auto& vector_id : vector_ids) {
        grpc_request.add_vector_ids(vector_id);
    }

    distributed::ShardDeleteResponse grpc_response;
    auto context = create_context(config_.write_timeout);

    grpc::Status status = connection->stub->DeleteFromShard(context.get(), grpc_request, &grpc_response);

    if (!status.ok()) {
        mark_worker_failed(worker_id);
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::RPC_ERROR, "RPC failed: " + status.error_message()));)
    }

    mark_worker_success(worker_id);
    return grpc_response.success();
#else
    logger_->warn("gRPC not enabled, delete operation not available");
    return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_IMPLEMENTED, "gRPC not enabled"));
#endif
}

// ============================================================================
// Health & Monitoring
// ============================================================================

Result<DistributedMasterClient::HealthCheckResponse> DistributedMasterClient::check_worker_health(
    const std::string& worker_id
) {
    logger_->debug("Checking health of worker: " + worker_id);

    auto connection_result = get_worker_connection(worker_id);
    if (!connection_result.has_value()) {
        return tl::unexpected(connection_result.error());
    }

#ifdef BUILD_WITH_GRPC
    auto connection = connection_result.value();

    distributed::HealthCheckRequest grpc_request;
    grpc_request.set_node_id(worker_id);
    grpc_request.set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());

    distributed::HealthCheckResponse grpc_response;
    auto context = create_context(config_.health_check_timeout);

    grpc::Status status = connection->stub->HealthCheck(context.get(), grpc_request, &grpc_response);

    if (!status.ok()) {
        mark_worker_failed(worker_id);

        HealthCheckResponse response;
        response.worker_id = worker_id;
        response.status = HealthStatus::UNHEALTHY;
        response.success = false;
        return response;
    }

    mark_worker_success(worker_id);

    HealthCheckResponse response;
    response.worker_id = grpc_response.node_id();
    response.status = static_cast<HealthStatus>(grpc_response.status());
    response.version = grpc_response.version();
    response.uptime_seconds = grpc_response.uptime_seconds();
    response.success = true;

    // TODO: Parse resource usage and shard statuses

    return response;
#else
    logger_->warn("gRPC not enabled, health check not available");
    return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_IMPLEMENTED, "gRPC not enabled"));
#endif
}

Result<std::unordered_map<std::string, DistributedMasterClient::HealthCheckResponse>>
DistributedMasterClient::check_all_workers_health() {
    logger_->info("Checking health of all workers");

    auto workers = get_connected_workers();
    std::unordered_map<std::string, HealthCheckResponse> results;

    for (const auto& worker_id : workers) {
        auto health_result = check_worker_health(worker_id);
        if (health_result.has_value()) {
            results[worker_id] = health_result.value();
        } else {
            logger_->warn("Failed to check health of worker: " + worker_id);
        }
    }

    return results;
}

Result<DistributedMasterClient::WorkerStatsResponse> DistributedMasterClient::get_worker_stats(
    const std::string& worker_id,
    bool include_shard_details
) {
    logger_->debug("Getting stats from worker: " + worker_id);

    auto connection_result = get_worker_connection(worker_id);
    if (!connection_result.has_value()) {
        return tl::unexpected(connection_result.error());
    }

#ifdef BUILD_WITH_GRPC
    auto connection = connection_result.value();

    distributed::WorkerStatsRequest grpc_request;
    grpc_request.set_node_id(worker_id);
    grpc_request.set_include_shard_details(include_shard_details);

    distributed::WorkerStatsResponse grpc_response;
    auto context = create_context(config_.default_timeout);

    grpc::Status status = connection->stub->GetWorkerStats(context.get(), grpc_request, &grpc_response);

    if (!status.ok()) {
        mark_worker_failed(worker_id);
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::RPC_ERROR, "RPC failed: " + status.error_message()));)
    }

    mark_worker_success(worker_id);

    WorkerStatsResponse response;
    response.worker_id = grpc_response.node_id();
    response.total_vectors = grpc_response.total_vectors();
    response.active_shards = grpc_response.active_shards();
    response.queries_processed = grpc_response.queries_processed();
    response.writes_processed = grpc_response.writes_processed();
    response.avg_query_latency_ms = grpc_response.avg_query_latency_ms();
    response.avg_write_latency_ms = grpc_response.avg_write_latency_ms();

    // TODO: Parse resource usage and shard stats

    return response;
#else
    logger_->warn("gRPC not enabled, stats operation not available");
    return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_IMPLEMENTED, "gRPC not enabled"));
#endif
}

// ============================================================================
// Shard Management
// ============================================================================

Result<bool> DistributedMasterClient::assign_shard(
    const std::string& worker_id,
    const std::string& shard_id,
    bool is_primary,
    const ShardConfig& config,
    const std::vector<uint8_t>& initial_data
) {
    logger_->info("Assigning shard " + shard_id + " to worker: " + worker_id);

    auto connection_result = get_worker_connection(worker_id);
    if (!connection_result.has_value()) {
        return tl::unexpected(connection_result.error());
    }

#ifdef BUILD_WITH_GRPC
    auto connection = connection_result.value();

    distributed::AssignShardRequest grpc_request;
    grpc_request.set_shard_id(shard_id);
    grpc_request.set_worker_id(worker_id);
    grpc_request.set_is_primary(is_primary);

    // TODO: Populate shard config
    if (!initial_data.empty()) {
        grpc_request.set_initial_data(initial_data.data(), initial_data.size());
    }

    distributed::AssignShardResponse grpc_response;
    auto context = create_context(config_.default_timeout);

    grpc::Status status = connection->stub->AssignShard(context.get(), grpc_request, &grpc_response);

    if (!status.ok()) {
        mark_worker_failed(worker_id);
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::RPC_ERROR, "RPC failed: " + status.error_message()));)
    }

    mark_worker_success(worker_id);
    return grpc_response.success();
#else
    logger_->warn("gRPC not enabled, assign shard operation not available");
    return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_IMPLEMENTED, "gRPC not enabled"));
#endif
}

Result<bool> DistributedMasterClient::remove_shard(
    const std::string& worker_id,
    const std::string& shard_id,
    bool force,
    const std::string& target_worker_id
) {
    logger_->info("Removing shard " + shard_id + " from worker: " + worker_id);

    auto connection_result = get_worker_connection(worker_id);
    if (!connection_result.has_value()) {
        return tl::unexpected(connection_result.error());
    }

#ifdef BUILD_WITH_GRPC
    auto connection = connection_result.value();

    distributed::RemoveShardRequest grpc_request;
    grpc_request.set_shard_id(shard_id);
    grpc_request.set_worker_id(worker_id);
    grpc_request.set_force(force);

    if (!target_worker_id.empty()) {
        grpc_request.set_transfer_to_worker(true);
        grpc_request.set_target_worker_id(target_worker_id);
    }

    distributed::RemoveShardResponse grpc_response;
    auto context = create_context(config_.default_timeout);

    grpc::Status status = connection->stub->RemoveShard(context.get(), grpc_request, &grpc_response);

    if (!status.ok()) {
        mark_worker_failed(worker_id);
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::RPC_ERROR, "RPC failed: " + status.error_message()));)
    }

    mark_worker_success(worker_id);
    return grpc_response.success();
#else
    logger_->warn("gRPC not enabled, remove shard operation not available");
    return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_IMPLEMENTED, "gRPC not enabled"));
#endif
}

Result<ShardInfo> DistributedMasterClient::get_shard_info(
    const std::string& worker_id,
    const std::string& shard_id,
    bool include_statistics
) {
    logger_->debug("Getting shard info for " + shard_id + " from worker: " + worker_id);

    auto connection_result = get_worker_connection(worker_id);
    if (!connection_result.has_value()) {
        return tl::unexpected(connection_result.error());
    }

#ifdef BUILD_WITH_GRPC
    auto connection = connection_result.value();

    distributed::ShardInfoRequest grpc_request;
    grpc_request.set_shard_id(shard_id);
    grpc_request.set_worker_id(worker_id);
    grpc_request.set_include_statistics(include_statistics);

    distributed::ShardInfoResponse grpc_response;
    auto context = create_context(config_.default_timeout);

    grpc::Status status = connection->stub->GetShardInfo(context.get(), grpc_request, &grpc_response);

    if (!status.ok()) {
        mark_worker_failed(worker_id);
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::RPC_ERROR, "RPC failed: " + status.error_message()));)
    }

    mark_worker_success(worker_id);

    ShardInfo info;
    info.shard_id = grpc_response.shard_id();
    info.is_primary = grpc_response.is_primary();
    info.state = static_cast<ShardState>(grpc_response.state());
    info.vector_count = grpc_response.vector_count();
    info.size_bytes = grpc_response.size_bytes();

    return info;
#else
    logger_->warn("gRPC not enabled, get shard info operation not available");
    return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_IMPLEMENTED, "gRPC not enabled"));
#endif
}

// ============================================================================
// Statistics
// ============================================================================

DistributedMasterClient::ClientStatistics DistributedMasterClient::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    ClientStatistics stats;
    stats.total_requests = total_requests_;
    stats.failed_requests = failed_requests_;

    if (total_requests_ > 0) {
        stats.failure_rate = static_cast<double>(failed_requests_) / total_requests_;
        stats.avg_request_time_ms = static_cast<double>(total_request_time_ms_) / total_requests_;
    } else {
        stats.failure_rate = 0.0;
        stats.avg_request_time_ms = 0.0;
    }

    std::lock_guard<std::mutex> conn_lock(connections_mutex_);
    stats.active_connections = 0;
    for (const auto& [_, connection] : worker_connections_) {
        if (connection->is_active) {
            stats.active_connections++;
        }
    }

    return stats;
}

void DistributedMasterClient::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    total_requests_ = 0;
    failed_requests_ = 0;
    total_request_time_ms_ = 0;
}

// ============================================================================
// Helper Methods
// ============================================================================

Result<std::shared_ptr<DistributedMasterClient::WorkerConnection>>
DistributedMasterClient::get_worker_connection(const std::string& worker_id) {
    std::lock_guard<std::mutex> lock(connections_mutex_);

    auto it = worker_connections_.find(worker_id);
    if (it == worker_connections_.end()) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NOT_FOUND, "Worker not found: " + worker_id));
    }

    if (!it->second->is_active) {
        return tl::make_unexpected(ErrorHandler::create_error(ErrorCode::NETWORK_ERROR, "Worker not active: " + worker_id));
    }

    return it->second;
}

#ifdef BUILD_WITH_GRPC
std::shared_ptr<grpc::Channel> DistributedMasterClient::create_channel(const std::string& host, int port) {
    std::string target = host + ":" + std::to_string(port);

    grpc::ChannelArguments args;
    args.SetMaxSendMessageSize(-1);  // Unlimited
    args.SetMaxReceiveMessageSize(-1);  // Unlimited

    if (config_.enable_compression) {
        args.SetCompressionAlgorithm(GRPC_COMPRESS_GZIP);
    }

    return grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), args);
}

std::unique_ptr<grpc::ClientContext> DistributedMasterClient::create_context(std::chrono::milliseconds timeout) {
    auto context = std::make_unique<grpc::ClientContext>();

    auto deadline = std::chrono::system_clock::now() + timeout;
    context->set_deadline(deadline);

    if (config_.enable_compression) {
        context->set_compression_algorithm(GRPC_COMPRESS_GZIP);
    }

    return context;
}
#endif

void DistributedMasterClient::record_request(bool success, int64_t duration_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    total_requests_++;
    if (!success) {
        failed_requests_++;
    }
    total_request_time_ms_ += duration_ms;
}

void DistributedMasterClient::mark_worker_failed(const std::string& worker_id) {
    std::lock_guard<std::mutex> lock(connections_mutex_);

    auto it = worker_connections_.find(worker_id);
    if (it != worker_connections_.end()) {
        it->second->consecutive_failures++;

        if (it->second->consecutive_failures >= 3) {
            it->second->is_active = false;
            logger_->warn("Worker marked as inactive due to failures: " + worker_id);
        }
    }
}

void DistributedMasterClient::mark_worker_success(const std::string& worker_id) {
    std::lock_guard<std::mutex> lock(connections_mutex_);

    auto it = worker_connections_.find(worker_id);
    if (it != worker_connections_.end()) {
        it->second->consecutive_failures = 0;
        it->second->last_success = std::chrono::steady_clock::now();

        if (!it->second->is_active) {
            it->second->is_active = true;
            logger_->info("Worker marked as active: " + worker_id);
        }
    }
}

// ============================================================================
// Replication Operations
// ============================================================================

Result<DistributedMasterClient::ReplicationResponse> DistributedMasterClient::replicate_data(
    const std::string& target_worker_id,
    const ReplicationRequest& request
) {
    auto start_time = std::chrono::steady_clock::now();

    auto connection_result = get_worker_connection(target_worker_id);
    if (!connection_result.has_value()) {
        record_request(false, 0);
        return tl::make_unexpected(connection_result.error());
    }

    auto connection = connection_result.value();

#ifdef BUILD_WITH_GRPC
    try {
        distributed::ReplicationRequest grpc_request;
        grpc_request.set_shard_id(request.shard_id);
        grpc_request.set_source_node_id(request.source_node_id);
        grpc_request.set_replication_type(
            request.replication_type == ReplicationType::FULL 
                ? distributed::REPLICATION_FULL 
                : distributed::REPLICATION_INCREMENTAL
        );
        grpc_request.set_from_version(request.from_version);
        grpc_request.set_to_version(request.to_version);

        // Add vectors to the request
        for (const auto& vector : request.vectors) {
            auto* proto_vector = grpc_request.add_vectors();
            proto_vector->set_vector_id(vector.id);
            for (const auto& val : vector.values) {
                proto_vector->add_values(val);
            }
            proto_vector->set_version(vector.version);
            proto_vector->set_timestamp(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count()
            );
            // Add metadata
            if (!vector.metadata.source.empty()) {
                (*proto_vector->mutable_metadata())["source"] = vector.metadata.source;
            }
            if (!vector.metadata.category.empty()) {
                (*proto_vector->mutable_metadata())["category"] = vector.metadata.category;
            }
        }

        auto context = create_context(config_.request_timeout);
        distributed::ReplicationResponse grpc_response;
        
        auto status = connection->stub->ReplicateData(context.get(), grpc_request, &grpc_response);

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time
        ).count();

        if (!status.ok()) {
            logger_->error("ReplicateData RPC failed for worker " + target_worker_id + ": " + status.error_message());
            record_request(false, duration);
            mark_worker_failed(target_worker_id);
            return tl::make_unexpected(ErrorHandler::create_error(
                ErrorCode::NETWORK_ERROR,
                "ReplicateData failed: " + status.error_message()
            ));
        }

        record_request(true, duration);
        mark_worker_success(target_worker_id);

        ReplicationResponse response;
        response.shard_id = grpc_response.shard_id();
        response.success = grpc_response.success();
        response.vectors_replicated = grpc_response.vectors_replicated();
        response.replication_lag_ms = grpc_response.replication_lag_ms();
        response.current_version = grpc_response.current_version();

        return response;

    } catch (const std::exception& e) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time
        ).count();
        record_request(false, duration);
        mark_worker_failed(target_worker_id);
        return tl::make_unexpected(ErrorHandler::create_error(
            ErrorCode::SERVICE_ERROR,
            "Exception in replicate_data: " + std::string(e.what())
        ));
    }
#else
    // Non-gRPC fallback: simulate replication
    logger_->debug("Simulating replicate_data to worker " + target_worker_id + " (no gRPC)");
    
    ReplicationResponse response;
    response.shard_id = request.shard_id;
    response.success = true;
    response.vectors_replicated = static_cast<int32_t>(request.vectors.size());
    response.replication_lag_ms = 0;
    response.current_version = request.to_version;
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time
    ).count();
    record_request(true, duration);
    
    return response;
#endif
}

Result<bool> DistributedMasterClient::sync_shard(
    const std::string& target_worker_id,
    const std::string& shard_id,
    const std::string& source_worker_id,
    int64_t target_version
) {
    auto connection_result = get_worker_connection(target_worker_id);
    if (!connection_result.has_value()) {
        return tl::make_unexpected(connection_result.error());
    }

#ifdef BUILD_WITH_GRPC
    try {
        distributed::SyncShardRequest grpc_request;
        grpc_request.set_shard_id(shard_id);
        grpc_request.set_source_node_id(source_worker_id);
        grpc_request.set_target_version(target_version);

        auto context = create_context(config_.request_timeout * 2);  // Double timeout for sync
        distributed::SyncShardResponse grpc_response;

        auto status = connection_result.value()->stub->SyncShard(context.get(), grpc_request, &grpc_response);

        if (!status.ok()) {
            logger_->error("SyncShard RPC failed: " + status.error_message());
            mark_worker_failed(target_worker_id);
            return tl::make_unexpected(ErrorHandler::create_error(
                ErrorCode::NETWORK_ERROR,
                "SyncShard failed: " + status.error_message()
            ));
        }

        mark_worker_success(target_worker_id);
        return grpc_response.success();

    } catch (const std::exception& e) {
        mark_worker_failed(target_worker_id);
        return tl::make_unexpected(ErrorHandler::create_error(
            ErrorCode::SERVICE_ERROR,
            "Exception in sync_shard: " + std::string(e.what())
        ));
    }
#else
    logger_->debug("Simulating sync_shard (no gRPC)");
    return true;
#endif
}

// ============================================================================
// Raft Consensus Operations
// ============================================================================

Result<DistributedMasterClient::VoteResponse> DistributedMasterClient::request_vote(
    const std::string& target_worker_id,
    const VoteRequest& request
) {
    auto connection_result = get_worker_connection(target_worker_id);
    if (!connection_result.has_value()) {
        return tl::make_unexpected(connection_result.error());
    }

#ifdef BUILD_WITH_GRPC
    try {
        distributed::RequestVoteRequest grpc_request;
        grpc_request.set_term(request.term);
        grpc_request.set_candidate_id(request.candidate_id);
        grpc_request.set_last_log_index(request.last_log_index);
        grpc_request.set_last_log_term(request.last_log_term);

        auto context = create_context(std::chrono::milliseconds(500));  // Short timeout for voting
        distributed::RequestVoteResponse grpc_response;

        auto status = connection_result.value()->stub->RequestVote(context.get(), grpc_request, &grpc_response);

        if (!status.ok()) {
            logger_->warn("RequestVote RPC failed for " + target_worker_id + ": " + status.error_message());
            mark_worker_failed(target_worker_id);
            return tl::make_unexpected(ErrorHandler::create_error(
                ErrorCode::NETWORK_ERROR,
                "RequestVote failed: " + status.error_message()
            ));
        }

        mark_worker_success(target_worker_id);

        VoteResponse response;
        response.term = grpc_response.term();
        response.vote_granted = grpc_response.vote_granted();
        response.voter_id = target_worker_id;

        return response;

    } catch (const std::exception& e) {
        mark_worker_failed(target_worker_id);
        return tl::make_unexpected(ErrorHandler::create_error(
            ErrorCode::SERVICE_ERROR,
            "Exception in request_vote: " + std::string(e.what())
        ));
    }
#else
    logger_->debug("Simulating request_vote (no gRPC)");
    VoteResponse response;
    response.term = request.term;
    response.vote_granted = true;  // Always grant in simulation
    response.voter_id = target_worker_id;
    return response;
#endif
}

Result<DistributedMasterClient::AppendEntriesResponse> DistributedMasterClient::append_entries(
    const std::string& target_worker_id,
    const AppendEntriesRequest& request
) {
    auto connection_result = get_worker_connection(target_worker_id);
    if (!connection_result.has_value()) {
        return tl::make_unexpected(connection_result.error());
    }

#ifdef BUILD_WITH_GRPC
    try {
        distributed::AppendEntriesRequest grpc_request;
        grpc_request.set_term(request.term);
        grpc_request.set_leader_id(request.leader_id);
        grpc_request.set_prev_log_index(request.prev_log_index);
        grpc_request.set_prev_log_term(request.prev_log_term);
        grpc_request.set_leader_commit(request.leader_commit_index);

        // Add log entries
        for (const auto& entry : request.entries) {
            auto* proto_entry = grpc_request.add_entries();
            proto_entry->set_term(entry.term);
            proto_entry->set_index(entry.index);
            proto_entry->set_command_type(entry.command_type);
            proto_entry->set_command_data(entry.command_data);
        }

        auto context = create_context(std::chrono::milliseconds(300));  // Short timeout for heartbeats
        distributed::AppendEntriesResponse grpc_response;

        auto status = connection_result.value()->stub->AppendEntries(context.get(), grpc_request, &grpc_response);

        if (!status.ok()) {
            logger_->debug("AppendEntries RPC failed for " + target_worker_id + ": " + status.error_message());
            mark_worker_failed(target_worker_id);
            return tl::make_unexpected(ErrorHandler::create_error(
                ErrorCode::NETWORK_ERROR,
                "AppendEntries failed: " + status.error_message()
            ));
        }

        mark_worker_success(target_worker_id);

        AppendEntriesResponse response;
        response.term = grpc_response.term();
        response.success = grpc_response.success();
        response.follower_id = target_worker_id;
        response.match_index = grpc_response.match_index();

        return response;

    } catch (const std::exception& e) {
        mark_worker_failed(target_worker_id);
        return tl::make_unexpected(ErrorHandler::create_error(
            ErrorCode::SERVICE_ERROR,
            "Exception in append_entries: " + std::string(e.what())
        ));
    }
#else
    logger_->debug("Simulating append_entries (no gRPC)");
    AppendEntriesResponse response;
    response.term = request.term;
    response.success = true;
    response.follower_id = target_worker_id;
    response.match_index = request.prev_log_index + static_cast<int64_t>(request.entries.size());
    return response;
#endif
}

Result<DistributedMasterClient::InstallSnapshotResponse> DistributedMasterClient::install_snapshot(
    const std::string& target_worker_id,
    const InstallSnapshotRequest& request
) {
    auto connection_result = get_worker_connection(target_worker_id);
    if (!connection_result.has_value()) {
        return tl::make_unexpected(connection_result.error());
    }

#ifdef BUILD_WITH_GRPC
    try {
        distributed::InstallSnapshotRequest grpc_request;
        grpc_request.set_term(request.term);
        grpc_request.set_leader_id(request.leader_id);
        grpc_request.set_last_included_index(request.last_included_index);
        grpc_request.set_last_included_term(request.last_included_term);
        grpc_request.set_offset(request.offset);
        grpc_request.set_data(request.data.data(), request.data.size());
        grpc_request.set_done(request.done);

        auto context = create_context(config_.default_timeout);
        distributed::InstallSnapshotResponse grpc_response;

        auto status = connection_result.value()->stub->InstallSnapshot(context.get(), grpc_request, &grpc_response);

        if (!status.ok()) {
            logger_->debug("InstallSnapshot RPC failed for " + target_worker_id + ": " + status.error_message());
            mark_worker_failed(target_worker_id);
            return tl::make_unexpected(ErrorHandler::create_error(
                ErrorCode::NETWORK_ERROR,
                "InstallSnapshot failed: " + status.error_message()
            ));
        }

        mark_worker_success(target_worker_id);

        InstallSnapshotResponse response;
        response.term = grpc_response.term();
        response.success = grpc_response.success();
        response.follower_id = grpc_response.follower_id();

        logger_->debug("InstallSnapshot to " + target_worker_id + " completed, success=" + 
                      std::to_string(response.success));

        return response;

    } catch (const std::exception& e) {
        mark_worker_failed(target_worker_id);
        return tl::make_unexpected(ErrorHandler::create_error(
            ErrorCode::SERVICE_ERROR,
            "Exception in install_snapshot: " + std::string(e.what())
        ));
    }
#else
    logger_->debug("Simulating install_snapshot (no gRPC)");
    InstallSnapshotResponse response;
    response.term = request.term;
    response.success = true;
    response.follower_id = target_worker_id;
    return response;
#endif
}

} // namespace jadevectordb
