#include "replication_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>
#include <chrono>

namespace jadevectordb {

ReplicationService::ReplicationService() {
    logger_ = logging::LoggerManager::get_logger("ReplicationService");
}

bool ReplicationService::initialize(const ReplicationConfig& config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (!validate_config(config)) {
            LOG_ERROR(logger_, "Invalid replication configuration provided");
            return false;
        }
        
        config_ = config;
        
        LOG_INFO(logger_, "ReplicationService initialized with default replication factor: " + 
                std::to_string(config_.default_replication_factor) + 
                ", synchronous replication: " + (config_.synchronous_replication ? "enabled" : "disabled"));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in ReplicationService::initialize: " + std::string(e.what()));
        return false;
    }
}

Result<bool> ReplicationService::replicate_vector(const Vector& vector, const Database& database) {
    try {
        LOG_DEBUG(logger_, "Replicating vector " + vector.id + " for database " + database.databaseId);
        
        // Get the replication factor for this database
        int replication_factor = get_replication_factor_for_db(database.databaseId);
        if (replication_factor <= 1) {
            LOG_DEBUG(logger_, "Replication factor is " + std::to_string(replication_factor) + ", skipping replication");
            return true;
        }
        
        // Select target nodes for replication based on current cluster state
        auto target_nodes_result = select_replica_nodes(vector.id, replication_factor);
        if (!target_nodes_result.has_value()) {
            LOG_ERROR(logger_, "Failed to select replica nodes for vector " + vector.id);
            return target_nodes_result;
        }
        
        auto target_nodes = target_nodes_result.value();
        if (target_nodes.empty()) {
            LOG_WARN(logger_, "No target nodes selected for replication of vector " + vector.id);
            return true;
        }
        
        // Send replication request to target nodes
        auto replication_result = send_replication_request(vector, target_nodes);
        if (!replication_result.has_value()) {
            LOG_ERROR(logger_, "Failed to replicate vector " + vector.id + " to target nodes");
            return replication_result;
        }
        
        // Update replication status
        update_replication_status(vector.id, database.databaseId, target_nodes);
        
        LOG_DEBUG(logger_, "Successfully replicated vector " + vector.id + " to " + 
                 std::to_string(target_nodes.size()) + " nodes");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in replicate_vector: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to replicate vector: " + std::string(e.what()));
    }
}

Result<bool> ReplicationService::replicate_vector_to_nodes(const Vector& vector, 
                                                         const std::vector<std::string>& target_nodes) {
    try {
        LOG_DEBUG(logger_, "Replicating vector " + vector.id + " to specific nodes");
        
        if (target_nodes.empty()) {
            LOG_WARN(logger_, "No target nodes specified for replication of vector " + vector.id);
            return true;
        }
        
        // Send replication request to specified target nodes
        auto replication_result = send_replication_request(vector, target_nodes);
        if (!replication_result.has_value()) {
            LOG_ERROR(logger_, "Failed to replicate vector " + vector.id + " to specified nodes");
            return replication_result;
        }
        
        LOG_DEBUG(logger_, "Successfully replicated vector " + vector.id + " to " + 
                 std::to_string(target_nodes.size()) + " specified nodes");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in replicate_vector_to_nodes: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to replicate vector to nodes: " + std::string(e.what()));
    }
}

Result<bool> ReplicationService::update_and_replicate(const Vector& updated_vector, 
                                                    const Database& database) {
    try {
        LOG_DEBUG(logger_, "Updating and replicating vector " + updated_vector.id);
        
        // First, apply the update locally
        auto update_result = apply_replicated_vector(updated_vector);
        if (!update_result.has_value()) {
            LOG_ERROR(logger_, "Failed to apply update locally for vector " + updated_vector.id);
            return update_result;
        }
        
        // Then replicate the update to other nodes
        auto replication_result = replicate_vector(updated_vector, database);
        if (!replication_result.has_value()) {
            LOG_ERROR(logger_, "Failed to replicate update for vector " + updated_vector.id);
            return replication_result;
        }
        
        LOG_DEBUG(logger_, "Successfully updated and replicated vector " + updated_vector.id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_and_replicate: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update and replicate vector: " + std::string(e.what()));
    }
}

Result<bool> ReplicationService::delete_and_replicate(const std::string& vector_id, 
                                                    const Database& database) {
    try {
        LOG_DEBUG(logger_, "Deleting and replicating deletion of vector " + vector_id);
        
        // First, delete locally
        auto delete_result = apply_replicated_vector(Vector{vector_id, {}}); // Empty vector to indicate deletion
        if (!delete_result.has_value()) {
            LOG_ERROR(logger_, "Failed to delete vector locally: " + vector_id);
            return delete_result;
        }
        
        // Then replicate the deletion to other nodes
        // For deletion, we create a special vector with empty values to indicate deletion
        Vector deletion_vector;
        deletion_vector.id = vector_id;
        deletion_vector.values = {}; // Empty values indicates deletion
        
        auto replication_result = replicate_vector(deletion_vector, database);
        if (!replication_result.has_value()) {
            LOG_ERROR(logger_, "Failed to replicate deletion of vector: " + vector_id);
            return replication_result;
        }
        
        LOG_DEBUG(logger_, "Successfully deleted and replicated deletion of vector " + vector_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in delete_and_replicate: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to delete and replicate vector: " + std::string(e.what()));
    }
}

Result<ReplicationStatus> ReplicationService::get_replication_status(const std::string& vector_id) const {
    try {
        std::lock_guard<std::mutex> lock(status_mutex_);
        
        auto it = replication_status_.find(vector_id);
        if (it != replication_status_.end()) {
            return it->second;
        }
        
        LOG_WARN(logger_, "Replication status not found for vector: " + vector_id);
        RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Replication status not found for vector: " + vector_id);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_replication_status: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get replication status: " + std::string(e.what()));
    }
}

Result<bool> ReplicationService::is_fully_replicated(const std::string& vector_id) const {
    try {
        auto status_result = get_replication_status(vector_id);
        if (!status_result.has_value()) {
            LOG_WARN(logger_, "Failed to get replication status for vector: " + vector_id);
            return status_result;
        }
        
        auto status = status_result.value();
        bool is_fully_replicated = static_cast<int>(status.replica_nodes.size()) >= status.replication_factor;
        
        LOG_DEBUG(logger_, "Vector " + vector_id + " is " + 
                 (is_fully_replicated ? "fully" : "not fully") + " replicated. " +
                 "Replicas: " + std::to_string(status.replica_nodes.size()) + "/" + 
                 std::to_string(status.replication_factor));
        
        return is_fully_replicated;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in is_fully_replicated: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check replication status: " + std::string(e.what()));
    }
}

Result<std::vector<std::string>> ReplicationService::get_replica_nodes(const std::string& vector_id) const {
    try {
        auto status_result = get_replication_status(vector_id);
        if (!status_result.has_value()) {
            LOG_WARN(logger_, "Failed to get replication status for vector: " + vector_id);
            return status_result;
        }
        
        auto status = status_result.value();
        return status.replica_nodes;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_replica_nodes: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get replica nodes: " + std::string(e.what()));
    }
}

Result<bool> ReplicationService::process_pending_replications() {
    try {
        LOG_INFO(logger_, "Processing pending replications");
        
        std::lock_guard<std::mutex> lock(status_mutex_);
        int processed_count = 0;
        
        for (auto& entry : replication_status_) {
            auto& status = entry.second;
            
            // Process any pending nodes
            for (const auto& node_id : status.pending_nodes) {
                // In a real implementation, we would actually send the replication request
                // For now, we'll just simulate success
                status.replica_nodes.push_back(node_id);
                processed_count++;
                
                LOG_DEBUG(logger_, "Processed pending replication for vector " + status.vector_id + 
                         " to node " + node_id);
            }
            
            // Clear pending nodes
            status.pending_nodes.clear();
        }
        
        LOG_INFO(logger_, "Processed " + std::to_string(processed_count) + " pending replications");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in process_pending_replications: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to process pending replications: " + std::string(e.what()));
    }
}

Result<bool> ReplicationService::check_replication_health(const std::string& database_id) const {
    try {
        LOG_DEBUG(logger_, "Checking replication health for database: " + database_id);
        
        std::lock_guard<std::mutex> lock(status_mutex_);
        int healthy_replicas = 0;
        int total_replicas = 0;
        
        // Check replication status for all vectors in this database
        for (const auto& entry : replication_status_) {
            const auto& status = entry.second;
            if (status.database_id == database_id) {
                total_replicas += status.replication_factor;
                healthy_replicas += status.replica_nodes.size();
            }
        }
        
        // Simple health check: at least 80% of expected replicas should be healthy
        if (total_replicas > 0) {
            double health_ratio = static_cast<double>(healthy_replicas) / total_replicas;
            bool is_healthy = health_ratio >= 0.8;
            
            LOG_DEBUG(logger_, "Replication health for database " + database_id + ": " + 
                     (is_healthy ? "healthy" : "unhealthy") + 
                     " (" + std::to_string(healthy_replicas) + "/" + 
                     std::to_string(total_replicas) + " replicas)");
            
            return is_healthy;
        }
        
        LOG_DEBUG(logger_, "No replication data found for database: " + database_id);
        return true; // No data to replicate, so considered healthy
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in check_replication_health: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check replication health: " + std::string(e.what()));
    }
}

Result<std::unordered_map<std::string, int>> ReplicationService::get_replication_stats() const {
    try {
        std::lock_guard<std::mutex> lock(status_mutex_);
        
        std::unordered_map<std::string, int> stats;
        stats["total_vectors"] = replication_status_.size();
        
        int total_replicas = 0;
        int expected_replicas = 0;
        
        for (const auto& entry : replication_status_) {
            const auto& status = entry.second;
            total_replicas += status.replica_nodes.size();
            expected_replicas += status.replication_factor;
        }
        
        stats["actual_replicas"] = total_replicas;
        stats["expected_replicas"] = expected_replicas;
        
        if (expected_replicas > 0) {
            stats["replication_ratio"] = static_cast<int>((static_cast<double>(total_replicas) / expected_replicas) * 100);
        } else {
            stats["replication_ratio"] = 100;
        }
        
        LOG_DEBUG(logger_, "Replication stats: total_vectors=" + std::to_string(stats["total_vectors"]) + 
                 ", actual_replicas=" + std::to_string(stats["actual_replicas"]) + 
                 ", expected_replicas=" + std::to_string(stats["expected_replicas"]) + 
                 ", replication_ratio=" + std::to_string(stats["replication_ratio"]) + "%");
        
        return stats;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_replication_stats: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get replication stats: " + std::string(e.what()));
    }
}

Result<bool> ReplicationService::handle_node_failure(const std::string& failed_node_id) {
    try {
        LOG_WARN(logger_, "Handling replication for failed node: " + failed_node_id);
        
        std::lock_guard<std::mutex> lock(status_mutex_);
        int re_replicated_count = 0;
        
        // For each vector that had replicas on the failed node, re-replicate to other nodes
        for (auto& entry : replication_status_) {
            auto& status = entry.second;
            
            // Check if this vector had replicas on the failed node
            auto it = std::find(status.replica_nodes.begin(), status.replica_nodes.end(), failed_node_id);
            if (it != status.replica_nodes.end()) {
                // Remove the failed node from replica list
                status.replica_nodes.erase(it);
                
                // Check if we need to re-replicate to maintain the replication factor
                int missing_replicas = status.replication_factor - static_cast<int>(status.replica_nodes.size());
                if (missing_replicas > 0) {
                    // Select new nodes for replication
                    auto new_nodes_result = select_replica_nodes(status.vector_id, missing_replicas);
                    if (new_nodes_result.has_value()) {
                        auto new_nodes = new_nodes_result.value();
                        
                        // Add new nodes to pending list
                        status.pending_nodes.insert(status.pending_nodes.end(), new_nodes.begin(), new_nodes.end());
                        re_replicated_count += new_nodes.size();
                        
                        LOG_DEBUG(logger_, "Scheduled re-replication for vector " + status.vector_id + 
                                 " to " + std::to_string(new_nodes.size()) + " new nodes");
                    } else {
                        LOG_WARN(logger_, "Failed to select new nodes for re-replication of vector " + 
                                status.vector_id);
                    }
                }
            }
        }
        
        LOG_INFO(logger_, "Handled node failure for " + failed_node_id + 
                ", scheduled re-replication for " + std::to_string(re_replicated_count) + " vectors");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_node_failure: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to handle node failure: " + std::string(e.what()));
    }
}

Result<bool> ReplicationService::add_node_and_replicate(const std::string& new_node_id) {
    try {
        LOG_INFO(logger_, "Adding new node and replicating data to it: " + new_node_id);
        
        // Add the new node to the cluster
        // In a real implementation, this would involve cluster membership changes
        
        std::lock_guard<std::mutex> lock(status_mutex_);
        int replicated_count = 0;
        
        // For demonstration, we'll replicate some data to the new node
        // In a real implementation, this would be more sophisticated
        for (const auto& entry : replication_status_) {
            const auto& status = entry.second;
            
            // Simple strategy: replicate a subset of data to the new node
            // In reality, this would be based on load balancing and sharding
            std::hash<std::string> hasher;
            if (hasher(status.vector_id) % 10 == 0) { // Replicate 10% of data for demo
                status.pending_nodes.push_back(new_node_id);
                replicated_count++;
            }
        }
        
        LOG_INFO(logger_, "Added new node " + new_node_id + 
                " and scheduled replication of " + std::to_string(replicated_count) + " vectors");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in add_node_and_replicate: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to add node and replicate: " + std::string(e.what()));
    }
}

Result<bool> ReplicationService::update_replication_config(const ReplicationConfig& new_config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (!validate_config(new_config)) {
            LOG_ERROR(logger_, "Invalid replication configuration provided for update");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid replication configuration");
        }
        
        config_ = new_config;
        
        LOG_INFO(logger_, "Updated replication configuration: default_replication_factor=" + 
                std::to_string(config_.default_replication_factor) + 
                ", synchronous_replication=" + (config_.synchronous_replication ? "enabled" : "disabled"));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_replication_config: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update replication configuration: " + std::string(e.what()));
    }
}

Result<bool> ReplicationService::force_replication_for_database(const std::string& database_id) {
    try {
        LOG_INFO(logger_, "Forcing replication for all data in database: " + database_id);
        
        std::lock_guard<std::mutex> lock(status_mutex_);
        int forced_count = 0;
        
        // Force re-replication of all vectors in this database
        for (auto& entry : replication_status_) {
            auto& status = entry.second;
            if (status.database_id == database_id) {
                // Clear current replica list and force re-replication
                status.replica_nodes.clear();
                
                // Select new nodes for replication
                auto new_nodes_result = select_replica_nodes(status.vector_id, status.replication_factor);
                if (new_nodes_result.has_value()) {
                    auto new_nodes = new_nodes_result.value();
                    
                    // Add new nodes to pending list
                    status.pending_nodes.insert(status.pending_nodes.end(), new_nodes.begin(), new_nodes.end());
                    forced_count++;
                    
                    LOG_DEBUG(logger_, "Forced re-replication for vector " + status.vector_id);
                } else {
                    LOG_WARN(logger_, "Failed to select nodes for forced replication of vector " + 
                            status.vector_id);
                }
            }
        }
        
        LOG_INFO(logger_, "Forced replication for " + std::to_string(forced_count) + 
                " vectors in database " + database_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in force_replication_for_database: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to force replication: " + std::string(e.what()));
    }
}

Result<std::vector<std::string>> ReplicationService::get_pending_replications() const {
    try {
        std::lock_guard<std::mutex> lock(status_mutex_);
        
        std::vector<std::string> pending_vectors;
        for (const auto& entry : replication_status_) {
            const auto& status = entry.second;
            if (!status.pending_nodes.empty()) {
                pending_vectors.push_back(status.vector_id);
            }
        }
        
        LOG_DEBUG(logger_, "Found " + std::to_string(pending_vectors.size()) + " vectors with pending replications");
        return pending_vectors;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_pending_replications: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get pending replications: " + std::string(e.what()));
    }
}

int ReplicationService::get_replication_factor_for_db(const std::string& database_id) const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    // In a real implementation, we might have per-database replication factors
    // For now, we'll just return the default
    return config_.default_replication_factor;
}

// Private methods

Result<std::vector<std::string>> ReplicationService::select_replica_nodes(const std::string& primary_node,
                                                                int replication_factor) const {
    // In a real implementation, this would select appropriate nodes based on:
    // - Current cluster topology
    // - Node capacity and load
    // - Failure domains
    // - Network proximity
    
    // For now, we'll return dummy node IDs
    std::vector<std::string> nodes;
    for (int i = 0; i < replication_factor; ++i) {
        nodes.push_back("replica_node_" + std::to_string(i));
    }
    
    return nodes;
}

Result<bool> ReplicationService::send_replication_request(const Vector& vector, 
                                                       const std::vector<std::string>& target_nodes) {
    try {
        LOG_DEBUG(logger_, "Sending replication request for vector " + vector.id + 
                 " to " + std::to_string(target_nodes.size()) + " nodes");
        
        // In a real implementation, this would:
        // 1. Serialize the vector data
        // 2. Send HTTP/gRPC requests to target nodes
        // 3. Handle responses and retries
        // 4. Update replication status
        
        // For now, we'll just simulate success
        for (const auto& node_id : target_nodes) {
            LOG_DEBUG(logger_, "Simulating replication to node: " + node_id);
            
            // Simulate network delay
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        LOG_DEBUG(logger_, "Replication request sent successfully for vector " + vector.id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in send_replication_request: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to send replication request: " + std::string(e.what()));
    }
}

void ReplicationService::update_replication_status(const std::string& vector_id, 
                                                 const std::string& database_id,
                                                 const std::vector<std::string>& replica_nodes) {
    try {
        std::lock_guard<std::mutex> lock(status_mutex_);
        
        ReplicationStatus status;
        status.vector_id = vector_id;
        status.database_id = database_id;
        status.replica_nodes = replica_nodes;
        status.replication_factor = get_replication_factor_for_db(database_id);
        status.last_updated = std::chrono::steady_clock::now();
        status.last_replicated = std::chrono::steady_clock::now();
        status.status = "replicating";
        
        replication_status_[vector_id] = status;
        
        LOG_DEBUG(logger_, "Updated replication status for vector " + vector_id + 
                 " with " + std::to_string(replica_nodes.size()) + " replicas");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_replication_status: " + std::string(e.what()));
    }
}

Result<bool> ReplicationService::apply_replicated_vector(const Vector& vector) {
    try {
        // In a real implementation, this would:
        // 1. Validate the replicated vector data
        // 2. Apply it to the local storage
        // 3. Update indexes and metadata
        
        LOG_DEBUG(logger_, "Applied replicated vector: " + vector.id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in apply_replicated_vector: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to apply replicated vector: " + std::string(e.what()));
    }
}

std::vector<std::string> ReplicationService::get_databases_on_node(const std::string& node_id) const {
    // In a real implementation, this would query the cluster state
    // For now, we'll return an empty list
    return {};
}

std::chrono::milliseconds ReplicationService::calculate_replication_lag(const std::string& vector_id) const {
    try {
        std::lock_guard<std::mutex> lock(status_mutex_);
        
        auto it = replication_status_.find(vector_id);
        if (it != replication_status_.end()) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.last_replicated);
            return elapsed;
        }
        
        return std::chrono::milliseconds(0);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in calculate_replication_lag: " + std::string(e.what()));
        return std::chrono::milliseconds(0);
    }
}

Result<bool> ReplicationService::perform_replication_by_strategy(const Vector& vector,
                                                              const std::vector<std::string>& target_nodes) {
    try {
        // Different replication strategies could be implemented here:
        // - Simple: replicate to all nodes at once
        // - Chain: replicate sequentially through a chain of nodes
        // - Star: replicate from a central node to all others
        
        // For now, we'll use simple replication
        return send_replication_request(vector, target_nodes);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in perform_replication_by_strategy: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to perform replication: " + std::string(e.what()));
    }
}

bool ReplicationService::validate_config(const ReplicationConfig& config) const {
    // Basic validation
    if (config.default_replication_factor < 1) {
        LOG_ERROR(logger_, "Invalid default replication factor: " + std::to_string(config.default_replication_factor));
        return false;
    }
    
    if (config.replication_timeout_ms < 0) {
        LOG_ERROR(logger_, "Invalid replication timeout: " + std::to_string(config.replication_timeout_ms));
        return false;
    }
    
    // Validate strategy
    if (!config.replication_strategy.empty() && 
        config.replication_strategy != "simple" && 
        config.replication_strategy != "chain" && 
        config.replication_strategy != "star") {
        LOG_ERROR(logger_, "Invalid replication strategy: " + config.replication_strategy);
        return false;
    }
    
    return true;
}

} // namespace jadevectordb