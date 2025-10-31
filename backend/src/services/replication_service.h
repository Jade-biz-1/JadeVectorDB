#ifndef JADEVECTORDB_REPLICATION_SERVICE_H
#define JADEVECTORDB_REPLICATION_SERVICE_H

#include "models/database.h"
#include "models/vector.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>

namespace jadevectordb {

// Represents replication status for a specific vector
struct ReplicationStatus {
    std::string vector_id;
    std::string database_id;
    std::vector<std::string> replica_nodes;  // Nodes that have replicas
    std::vector<std::string> pending_nodes;  // Nodes where replication is pending
    int replication_factor;                  // Expected number of replicas
    std::chrono::steady_clock::time_point last_updated;
    std::chrono::steady_clock::time_point last_replicated;
    std::string status;                      // "replicating", "completed", "failed", "pending"
    
    ReplicationStatus() : replication_factor(0) {}
    ReplicationStatus(const std::string& id, const std::string& db_id, int factor)
        : vector_id(id), database_id(db_id), replication_factor(factor), 
          last_updated(std::chrono::steady_clock::now()),
          last_replicated(std::chrono::steady_clock::now()),
          status("pending") {}
};

// Configuration for replication
struct ReplicationConfig {
    int default_replication_factor;      // Default number of replicas per vector
    bool synchronous_replication;        // Whether to wait for replicas to confirm writes
    int replication_timeout_ms;          // Timeout for replication operations
    std::string replication_strategy;    // "simple", "chain", "star", etc.
    bool enable_cross_region;            // Whether to allow cross-region replication
    std::vector<std::string> preferred_regions; // Preferred regions for replication
    
    ReplicationConfig() : 
        default_replication_factor(1), 
        synchronous_replication(false), 
        replication_timeout_ms(5000) {}
};

/**
 * @brief Service for managing data replication across distributed nodes
 * 
 * This service handles data replication to ensure durability and availability
 * of vector data across multiple nodes in the cluster.
 */
class ReplicationService {
private:
    std::shared_ptr<logging::Logger> logger_;
    ReplicationConfig config_;
    std::unordered_map<std::string, ReplicationStatus> replication_status_; // vector_id -> status
    std::unordered_map<std::string, std::vector<std::string>> db_replicas_; // database_id -> node_ids
    mutable std::mutex status_mutex_;
    mutable std::mutex config_mutex_;
    
public:
    explicit ReplicationService();
    ~ReplicationService() = default;
    
    // Initialize the replication service with configuration
    bool initialize(const ReplicationConfig& config);
    
    // Replicate a vector to multiple nodes
    Result<void> replicate_vector(const Vector& vector, const Database& database);
    
    // Replicate a vector to specific nodes
    Result<void> replicate_vector_to_nodes(const Vector& vector, 
                                         const std::vector<std::string>& target_nodes);
    
    // Update a vector and replicate the changes
    Result<void> update_and_replicate(const Vector& updated_vector, 
                                    const Database& database);
    
    // Delete a vector and its replicas
    Result<void> delete_and_replicate(const std::string& vector_id, 
                                    const Database& database);
    
    // Get replication status for a specific vector
    Result<ReplicationStatus> get_replication_status(const std::string& vector_id) const;
    
    // Check if a vector is fully replicated according to the policy
    Result<bool> is_fully_replicated(const std::string& vector_id) const;
    
    // Get all nodes that have replicas of a vector
    Result<std::vector<std::string>> get_replica_nodes(const std::string& vector_id) const;
    
    // Trigger replication for all pending operations
    Result<bool> process_pending_replications();
    
    // Check replication health for a database
    Result<bool> check_replication_health(const std::string& database_id) const;
    
    // Get replication statistics
    Result<std::unordered_map<std::string, int>> get_replication_stats() const;
    
    // Handle node failure and re-replicate data
    Result<void> handle_node_failure(const std::string& failed_node_id);
    
    // Add a new node and replicate data to it
    Result<void> add_node_and_replicate(const std::string& new_node_id);
    
    // Update replication configuration
    Result<bool> update_replication_config(const ReplicationConfig& new_config);
    
    // Force replication of all data in a database
    Result<bool> force_replication_for_database(const std::string& database_id);
    
    // Get all vectors that need replication
    Result<std::vector<std::string>> get_pending_replications() const;
    
    // Get the replication factor for a specific database
    int get_replication_factor_for_db(const std::string& database_id) const;

private:
    // Select target nodes for replication based on current cluster state
    Result<std::vector<std::string>> select_replica_nodes(const std::string& primary_node,
                                                int replication_factor) const;
    
    // Send vector data to target nodes for replication
    Result<bool> send_replication_request(const Vector& vector, 
                                        const std::vector<std::string>& target_nodes);
    
    // Update replication status after operation completes
    void update_replication_status(const std::string& vector_id, 
                                 const std::string& database_id, 
                                 const std::vector<std::string>& replica_nodes);
    
    // Validate replication configuration
    bool validate_config(const ReplicationConfig& config) const;
    
    // Apply vector to the local storage (used during replication receive)
    Result<bool> apply_replicated_vector(const Vector& vector);
    
    // Get all databases that exist on a specific node
    std::vector<std::string> get_databases_on_node(const std::string& node_id) const;
    
    // Calculate replication lag for monitoring
    std::chrono::milliseconds calculate_replication_lag(const std::string& vector_id) const;
    
    // Perform replication using the configured strategy
    Result<bool> perform_replication_by_strategy(const Vector& vector,
                                               const std::vector<std::string>& target_nodes);
};

} // namespace jadevectordb

#endif // JADEVECTORDB_REPLICATION_SERVICE_H