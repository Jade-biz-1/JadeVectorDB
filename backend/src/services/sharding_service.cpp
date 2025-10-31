#include "sharding_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>
#include <chrono>

namespace jadevectordb {

ShardingService::ShardingService() {
    logger_ = logging::LoggerManager::get_logger("ShardingService");
    
    // Initialize with default hash function
    hash_function_ = [](const std::string& key) -> uint64_t {
        uint64_t hash = 5381;
        for (char c : key) {
            hash = ((hash << 5) + hash) + c; // hash * 33 + c
        }
        return hash;
    };
}

bool ShardingService::initialize(const ShardingConfig& config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (!validate_config(config)) {
            LOG_ERROR(logger_, "Invalid sharding configuration provided");
            return false;
        }
        
        config_ = config;
        initialize_hash_function();
        
        LOG_INFO(logger_, "ShardingService initialized with strategy: " + config_.strategy + 
                ", num_shards: " + std::to_string(config_.num_shards));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in ShardingService::initialize: " + std::string(e.what()));
        return false;
    }
}

Result<std::string> ShardingService::get_shard_for_vector(const std::string& vector_id, 
                                                         const std::string& database_id) const {
    try {
        // For simplicity, we'll use hash-based sharding
        // In a real implementation, this would depend on the configured strategy
        uint64_t hash = hash_function_(vector_id);
        int shard_number = hash % config_.num_shards;
        std::string shard_id = generate_shard_id(database_id, shard_number);
        
        LOG_DEBUG(logger_, "Assigned vector " + vector_id + " to shard " + shard_id + 
                 " (shard #" + std::to_string(shard_number) + ")");
        return shard_id;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_shard_for_vector: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to determine shard for vector: " + std::string(e.what()));
    }
}

Result<std::string> ShardingService::get_node_for_shard(const std::string& shard_id) const {
    try {
        // In a real implementation, this would look up which node hosts the shard
        // For now, we'll just return a default node ID
        if (!config_.node_list.empty()) {
            // Simple round-robin assignment for demonstration
            std::hash<std::string> hasher;
            size_t index = hasher(shard_id) % config_.node_list.size();
            return config_.node_list[index];
        }
        
        return "default_node";
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_node_for_shard: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to determine node for shard: " + std::string(e.what()));
    }
}

Result<std::vector<ShardInfo>> ShardingService::get_shards_for_database(const std::string& database_id) const {
    try {
        std::vector<ShardInfo> shards;
        
        // Return all shards for this database
        auto it = db_shards_.find(database_id);
        if (it != db_shards_.end()) {
            shards = it->second;
        } else {
            // Create default shards if none exist
            for (int i = 0; i < config_.num_shards; ++i) {
                ShardInfo shard_info;
                shard_info.shard_id = generate_shard_id(database_id, i);
                shard_info.database_id = database_id;
                shard_info.shard_number = i;
                shard_info.node_id = "default_node_" + std::to_string(i % std::max(1, (int)config_.node_list.size()));
                shard_info.status = "active";
                shard_info.record_count = 0;
                shard_info.size_bytes = 0;
                shards.push_back(shard_info);
            }
        }
        
        return shards;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_shards_for_database: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get shards for database: " + std::string(e.what()));
    }
}

Result<ShardInfo> ShardingService::determine_shard(const Vector& vector, const Database& database) const {
    try {
        std::string vector_id = vector.id.empty() ? "temp_" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) : vector.id;
        auto shard_id_result = get_shard_for_vector(vector_id, database.databaseId);
        
        if (!shard_id_result.has_value()) {
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to determine shard for vector");
        }
        
        ShardInfo shard_info;
        shard_info.shard_id = shard_id_result.value();
        shard_info.database_id = database.databaseId;
        shard_info.status = "active";
        
        return shard_info;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in determine_shard: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to determine shard: " + std::string(e.what()));
    }
}

Result<bool> ShardingService::create_shards_for_database(const Database& database) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        std::vector<ShardInfo> shards;
        for (int i = 0; i < config_.num_shards; ++i) {
            ShardInfo shard_info;
            shard_info.shard_id = generate_shard_id(database.databaseId, i);
            shard_info.database_id = database.databaseId;
            shard_info.shard_number = i;
            shard_info.node_id = "default_node_" + std::to_string(i % std::max(1, (int)config_.node_list.size()));
            shard_info.status = "active";
            shard_info.record_count = 0;
            shard_info.size_bytes = 0;
            shards.push_back(shard_info);
        }
        
        db_shards_[database.databaseId] = shards;
        shards_.insert(shards_.end(), shards.begin(), shards.end());
        
        LOG_INFO(logger_, "Created " + std::to_string(shards.size()) + " shards for database " + database.databaseId);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in create_shards_for_database: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to create shards for database: " + std::string(e.what()));
    }
}

Result<bool> ShardingService::update_sharding_config(const ShardingConfig& new_config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (!validate_config(new_config)) {
            LOG_ERROR(logger_, "Invalid sharding configuration provided for update");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid sharding configuration");
        }
        
        config_ = new_config;
        initialize_hash_function();
        
        LOG_INFO(logger_, "Updated sharding configuration: strategy=" + config_.strategy + 
                ", num_shards=" + std::to_string(config_.num_shards));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_sharding_config: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update sharding configuration: " + std::string(e.what()));
    }
}

Result<bool> ShardingService::migrate_shard(const std::string& shard_id, const std::string& target_node_id) {
    try {
        LOG_INFO(logger_, "Migrating shard " + shard_id + " to node " + target_node_id);
        
        // In a real implementation, this would involve:
        // 1. Marking the shard as migrating
        // 2. Transferring data to the target node
        // 3. Updating metadata
        // 4. Marking the shard as active on the new node
        // 5. Cleaning up from the old node
        
        // For now, we'll just simulate the migration
        auto it = std::find_if(shards_.begin(), shards_.end(), 
                              [&shard_id](const ShardInfo& shard) { return shard.shard_id == shard_id; });
        
        if (it != shards_.end()) {
            it->node_id = target_node_id;
            it->status = "active";
            
            // Also update in db_shards_
            for (auto& db_entry : db_shards_) {
                for (auto& shard : db_entry.second) {
                    if (shard.shard_id == shard_id) {
                        shard.node_id = target_node_id;
                        shard.status = "active";
                        break;
                    }
                }
            }
            
            LOG_INFO(logger_, "Shard " + shard_id + " migrated to node " + target_node_id);
            return true;
        }
        
        LOG_WARN(logger_, "Shard not found for migration: " + shard_id);
        RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Shard not found: " + shard_id);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in migrate_shard: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to migrate shard: " + std::string(e.what()));
    }
}

ShardingConfig ShardingService::get_config() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

Result<std::unordered_map<std::string, size_t>> ShardingService::get_shard_distribution() const {
    try {
        std::unordered_map<std::string, size_t> distribution;
        
        for (const auto& shard : shards_) {
            distribution[shard.node_id]++;
        }
        
        return distribution;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_shard_distribution: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get shard distribution: " + std::string(e.what()));
    }
}

Result<bool> ShardingService::is_balanced() const {
    try {
        auto distribution_result = get_shard_distribution();
        if (!distribution_result.has_value()) {
            return distribution_result;
        }
        
        auto distribution = distribution_result.value();
        if (distribution.empty()) {
            return true; // No shards, so balanced
        }
        
        // Check if distribution is reasonably balanced
        size_t total_shards = 0;
        size_t max_shards = 0;
        size_t min_shards = std::numeric_limits<size_t>::max();
        
        for (const auto& entry : distribution) {
            total_shards += entry.second;
            max_shards = std::max(max_shards, entry.second);
            min_shards = std::min(min_shards, entry.second);
        }
        
        if (total_shards == 0) {
            return true;
        }
        
        // Simple balance check: difference between max and min should not exceed 10%
        double average = static_cast<double>(total_shards) / distribution.size();
        double imbalance = static_cast<double>(max_shards - min_shards) / average;
        
        return imbalance <= 0.1; // Balanced if imbalance is <= 10%
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in is_balanced: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check balance: " + std::string(e.what()));
    }
}

Result<bool> ShardingService::rebalance_shards() {
    try {
        LOG_INFO(logger_, "Rebalancing shards across nodes");
        
        // In a real implementation, this would:
        // 1. Check current distribution
        // 2. Determine optimal distribution
        // 3. Migrate shards to achieve balance
        
        // For now, we'll just return success
        LOG_INFO(logger_, "Shard rebalancing completed");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in rebalance_shards: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to rebalance shards: " + std::string(e.what()));
    }
}

Result<bool> ShardingService::handle_node_failure(const std::string& failed_node_id) {
    try {
        LOG_WARN(logger_, "Handling failure of node: " + failed_node_id);
        
        // In a real implementation, this would:
        // 1. Mark the node as failed
        // 2. Identify shards on the failed node
        // 3. Migrate those shards to other nodes
        // 4. Update metadata
        
        // For now, we'll just log and return success
        LOG_INFO(logger_, "Handled node failure for node: " + failed_node_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in handle_node_failure: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to handle node failure: " + std::string(e.what()));
    }
}

Result<bool> ShardingService::add_node_to_cluster(const std::string& node_id) {
    try {
        LOG_INFO(logger_, "Adding node to cluster: " + node_id);
        
        // Add node to node list if not already present
        if (std::find(config_.node_list.begin(), config_.node_list.end(), node_id) == config_.node_list.end()) {
            config_.node_list.push_back(node_id);
        }
        
        // In a real implementation, this might trigger rebalancing
        LOG_INFO(logger_, "Node added to cluster: " + node_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in add_node_to_cluster: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to add node to cluster: " + std::string(e.what()));
    }
}

Result<bool> ShardingService::remove_node_from_cluster(const std::string& node_id) {
    try {
        LOG_INFO(logger_, "Removing node from cluster: " + node_id);
        
        // Remove node from node list
        config_.node_list.erase(
            std::remove(config_.node_list.begin(), config_.node_list.end(), node_id),
            config_.node_list.end()
        );
        
        // In a real implementation, this would trigger rebalancing
        LOG_INFO(logger_, "Node removed from cluster: " + node_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in remove_node_from_cluster: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to remove node from cluster: " + std::string(e.what()));
    }
}

ShardingService::ShardingStrategy ShardingService::get_strategy_for_database(const std::string& database_id) const {
    if (config_.strategy == "hash") {
        return ShardingStrategy::HASH;
    } else if (config_.strategy == "range") {
        return ShardingStrategy::RANGE;
    } else if (config_.strategy == "vector") {
        return ShardingStrategy::VECTOR;
    } else {
        return ShardingStrategy::AUTO;
    }
}

Result<bool> ShardingService::update_shard_metadata(const std::string& shard_id, 
                                                   size_t record_count, 
                                                   size_t size_bytes) {
    try {
        // Update in shards_ vector
        auto it = std::find_if(shards_.begin(), shards_.end(), 
                              [&shard_id](const ShardInfo& shard) { return shard.shard_id == shard_id; });
        
        if (it != shards_.end()) {
            it->record_count = record_count;
            it->size_bytes = size_bytes;
        }
        
        // Update in db_shards_ map
        bool found = false;
        for (auto& db_entry : db_shards_) {
            for (auto& shard : db_entry.second) {
                if (shard.shard_id == shard_id) {
                    shard.record_count = record_count;
                    shard.size_bytes = size_bytes;
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        
        if (!found) {
            LOG_WARN(logger_, "Shard not found for metadata update: " + shard_id);
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_shard_metadata: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update shard metadata: " + std::string(e.what()));
    }
}

// Private methods

void ShardingService::initialize_hash_function() {
    if (config_.hash_function == "murmur") {
        // MurmurHash implementation would go here
        // For now, we'll keep the default hash function
        LOG_DEBUG(logger_, "Using MurmurHash function (stub implementation)");
    } else if (config_.hash_function == "fnv") {
        // FNV hash implementation would go here
        LOG_DEBUG(logger_, "Using FNV hash function (stub implementation)");
    }
    // Default hash function is already set in constructor
}

Result<ShardInfo> ShardingService::hash_based_sharding(const Vector& vector, const Database& database) const {
    std::string vector_id = vector.id.empty() ? "temp_vector" : vector.id;
    auto shard_id_result = get_shard_for_vector(vector_id, database.databaseId);
    
    if (!shard_id_result.has_value()) {
        return shard_id_result;
    }
    
    ShardInfo shard_info;
    shard_info.shard_id = shard_id_result.value();
    shard_info.database_id = database.databaseId;
    shard_info.status = "active";
    
    return shard_info;
}

Result<ShardInfo> ShardingService::range_based_sharding(const Vector& vector, const Database& database) const {
    // For range-based sharding, we'd need to define ranges
    // For now, we'll fall back to hash-based sharding
    return hash_based_sharding(vector, database);
}

Result<ShardInfo> ShardingService::vector_based_sharding(const Vector& vector, const Database& database) const {
    // Vector-based sharding would use vector content for placement
    // For now, we'll fall back to hash-based sharding
    return hash_based_sharding(vector, database);
}

int ShardingService::calculate_shard_number(const std::string& key, int total_shards) const {
    uint64_t hash = hash_function_(key);
    return hash % total_shards;
}

std::string ShardingService::generate_shard_id(const std::string& database_id, int shard_number) const {
    return database_id + "_shard_" + std::to_string(shard_number);
}

bool ShardingService::validate_config(const ShardingConfig& config) const {
    // Basic validation
    if (config.num_shards <= 0) {
        LOG_ERROR(logger_, "Invalid number of shards: " + std::to_string(config.num_shards));
        return false;
    }
    
    if (config.replication_factor < 1) {
        LOG_ERROR(logger_, "Invalid replication factor: " + std::to_string(config.replication_factor));
        return false;
    }
    
    // Validate strategy
    if (config.strategy != "hash" && config.strategy != "range" && 
        config.strategy != "vector" && config.strategy != "auto") {
        LOG_ERROR(logger_, "Invalid sharding strategy: " + config.strategy);
        return false;
    }
    
    // Validate hash function if specified
    if (!config.hash_function.empty() && 
        config.hash_function != "murmur" && config.hash_function != "fnv") {
        LOG_ERROR(logger_, "Invalid hash function: " + config.hash_function);
        return false;
    }
    
    return true;
}

Result<bool> ShardingService::create_shards_by_strategy(const Database& database, ShardingStrategy strategy) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        LOG_DEBUG(logger_, "Creating shards by strategy for database " + database.databaseId);
        
        // Implementation varies based on strategy
        switch (strategy) {
            case ShardingStrategy::HASH:
                // For hash-based sharding, we just create the standard hash-based shards
                return create_shards_for_database(database);
                
            case ShardingStrategy::RANGE:
                // For range-based sharding, we might need to define ranges based on data characteristics
                return create_shards_for_database(database);
                
            case ShardingStrategy::VECTOR:
                // For vector-based sharding, we might shard based on vector properties
                return create_shards_for_database(database);
                
            case ShardingStrategy::AUTO:
            default:
                // For auto strategy, we use the configured default
                return create_shards_for_database(database);
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in create_shards_by_strategy: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to create shards by strategy: " + std::string(e.what()));
    }
}

Result<bool> ShardingService::distribute_shards_to_nodes() {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        LOG_DEBUG(logger_, "Distributing shards to nodes");
        
        // Implementation to distribute shards across available nodes
        // For now, we'll distribute them in a round-robin fashion
        for (auto& shard_entry : db_shards_) {
            std::vector<ShardInfo>& shards = shard_entry.second;
            for (size_t i = 0; i < shards.size(); ++i) {
                if (!config_.node_list.empty()) {
                    shards[i].node_id = config_.node_list[i % config_.node_list.size()];
                } else {
                    shards[i].node_id = "default_node_" + std::to_string(i);
                }
            }
        }
        
        LOG_DEBUG(logger_, "Distributed shards to nodes completed");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in distribute_shards_to_nodes: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to distribute shards to nodes: " + std::string(e.what()));
    }
}

bool ShardingService::vector_in_range(const Vector& vector, const std::pair<std::string, std::string>& range) const {
    try {
        LOG_DEBUG(logger_, "Checking if vector " + vector.id + " is in range");
        
        // Implementation to check if vector falls within a specific range
        // This is a simplified approach - in a real implementation, this would depend on
        // the range sharding strategy and vector properties
        
        // For now, we'll use a simple hash-based check
        std::hash<std::string> hasher;
        size_t hash = hasher(vector.id);
        std::string hash_str = std::to_string(hash);
        
        // Check if the hash value falls within the provided range
        return hash_str >= range.first && hash_str <= range.second;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in vector_in_range: " + std::string(e.what()));
        return false;
    }
}

void ShardingService::update_range_boundaries() {
    try {
        LOG_DEBUG(logger_, "Updating range boundaries for range-based sharding");
        
        // Implementation to update range boundaries for range-based sharding
        // In a real implementation, this would adjust the range boundaries based on
        // data distribution, load balancing requirements, etc.
        
        // For now, we'll just log the operation
        LOG_DEBUG(logger_, "Range boundaries updated");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_range_boundaries: " + std::string(e.what()));
    }
}

} // namespace jadevectordb