#ifndef JADEVECTORDB_DISTRIBUTED_SERVICE_MANAGER_H
#define JADEVECTORDB_DISTRIBUTED_SERVICE_MANAGER_H

#include "sharding_service.h"
#include "replication_service.h"
#include "query_router.h"
#include "cluster_service.h"
#include "security_audit_logger.h"
#include "performance_benchmark.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include "lib/config.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace jadevectordb {

// Configuration for distributed services
struct DistributedConfig {
    ShardingConfig sharding_config;
    ReplicationConfig replication_config;
    RoutingConfig routing_config;
    std::string cluster_host;
    int cluster_port;
    std::vector<std::string> seed_nodes; // List of seed nodes to join the cluster
    bool enable_sharding;
    bool enable_replication;
    bool enable_clustering;
    
    DistributedConfig() : cluster_port(0), enable_sharding(true), 
                         enable_replication(true), enable_clustering(true) {}
};

/**
 * @brief Manager for coordinating distributed services in the vector database
 * 
 * This service manages the lifecycle and coordination of all distributed components:
 * - ShardingService for data distribution
 * - ReplicationService for data durability
 * - QueryRouter for request routing
 * - ClusterService for cluster membership and coordination
 */
class DistributedServiceManager {
private:
    std::shared_ptr<logging::Logger> logger_;
    DistributedConfig config_;
    
    // Distributed services
    std::shared_ptr<ShardingService> sharding_service_;
    std::shared_ptr<ReplicationService> replication_service_;
    std::shared_ptr<QueryRouter> query_router_;
    std::unique_ptr<ClusterService> cluster_service_;
    
    // Additional services
    std::shared_ptr<SecurityAuditLogger> security_audit_logger_;
    std::shared_ptr<PerformanceBenchmark> performance_benchmark_;
    
    // State management
    mutable std::mutex config_mutex_;
    mutable std::mutex service_mutex_;
    bool initialized_;
    bool running_;

public:
    explicit DistributedServiceManager();
    ~DistributedServiceManager();
    
    // Initialize all distributed services with configuration
    Result<bool> initialize(const DistributedConfig& config);
    
    // Start all distributed services
    Result<bool> start();
    
    // Stop all distributed services
    Result<bool> stop();
    
    // Check if distributed services are initialized
    bool is_initialized() const;
    
    // Check if distributed services are running
    bool is_running() const;
    
    // Get individual services
    std::shared_ptr<ShardingService> get_sharding_service() const;
    std::shared_ptr<ReplicationService> get_replication_service() const;
    std::shared_ptr<QueryRouter> get_query_router() const;
    ClusterService* get_cluster_service() const;
    std::shared_ptr<SecurityAuditLogger> get_security_audit_logger() const;
    std::shared_ptr<PerformanceBenchmark> get_performance_benchmark() const;
    
    // Cluster management
    Result<bool> join_cluster(const std::string& seed_node_host, int seed_node_port);
    Result<bool> leave_cluster();
    Result<ClusterState> get_cluster_state() const;
    Result<bool> is_cluster_healthy() const;
    
    // Node management
    Result<bool> add_node_to_cluster(const std::string& node_id);
    Result<bool> remove_node_from_cluster(const std::string& node_id);
    Result<bool> handle_node_failure(const std::string& failed_node_id);
    
    // Configuration management
    Result<bool> update_distributed_config(const DistributedConfig& new_config);
    DistributedConfig get_config() const;
    
    // Sharding operations
    Result<bool> create_shards_for_database(const Database& database);
    Result<std::string> get_shard_for_vector(const std::string& vector_id, 
                                          const std::string& database_id) const;
    Result<std::string> get_node_for_shard(const std::string& shard_id) const;
    
    // Replication operations
    Result<bool> replicate_vector(const Vector& vector, const Database& database);
    Result<bool> is_vector_fully_replicated(const std::string& vector_id) const;
    
    // Routing operations
    Result<RouteInfo> route_operation(const std::string& database_id,
                                   const std::string& operation_type,
                                   const std::string& operation_key = "") const;
    
    // Health and monitoring
    Result<std::unordered_map<std::string, std::string>> get_distributed_stats() const;
    Result<bool> check_distributed_health() const;
    
    // Utility methods
    Result<bool> rebalance_shards();
    Result<bool> force_replication_for_database(const std::string& database_id);
    
    // Performance and security methods
    Result<BenchmarkResult> run_distributed_benchmark(const BenchmarkConfig& config,
                                                     const std::function<Result<BenchmarkOperationResult>()>& operation_func);
    Result<bool> audit_security_event(const SecurityEvent& event);

private:
    // Initialize individual services
    Result<bool> initialize_sharding_service();
    Result<bool> initialize_replication_service();
    Result<bool> initialize_query_router();
    Result<bool> initialize_cluster_service();
    
    // Start individual services
    Result<bool> start_sharding_service();
    Result<bool> start_replication_service();
    Result<bool> start_query_router();
    Result<bool> start_cluster_service();
    
    // Stop individual services
    Result<bool> stop_sharding_service();
    Result<bool> stop_replication_service();
    Result<bool> stop_query_router();
    Result<bool> stop_cluster_service();
    
    // Validate configuration
    bool validate_config(const DistributedConfig& config) const;
    
    // Coordinate services
    Result<bool> coordinate_services();
    
    // Handle service failures
    void handle_service_failure(const std::string& service_name, const std::string& error_message);
    
    // Update service dependencies
    Result<bool> update_service_dependencies();
};

} // namespace jadevectordb

#endif // JADEVECTORDB_DISTRIBUTED_SERVICE_MANAGER_H