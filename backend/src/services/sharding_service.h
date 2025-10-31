#ifndef JADEVECTORDB_SHARDING_SERVICE_H
#define JADEVECTORDB_SHARDING_SERVICE_H

#include "models/database.h"
#include "models/vector.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <mutex>

namespace jadevectordb {

// Represents a shard in the distributed system
struct ShardInfo {
    std::string shard_id;
    std::string database_id;
    int shard_number;           // Sequential number of the shard
    std::string node_id;       // ID of the node that hosts this shard
    std::string status;        // "active", "migrating", "offline", etc.
    size_t record_count;       // Number of vectors in this shard
    size_t size_bytes;         // Size of the shard in bytes
    std::string hash_range_start; // For range-based sharding
    std::string hash_range_end;   // For range-based sharding
    
    ShardInfo() : shard_number(0), record_count(0), size_bytes(0) {}
    ShardInfo(const std::string& id, const std::string& db_id, int num)
        : shard_id(id), database_id(db_id), shard_number(num), status("active"), 
          record_count(0), size_bytes(0) {}
};

// Configuration for sharding strategy
struct ShardingConfig {
    std::string strategy;          // "hash", "range", "vector", "auto"
    int num_shards;               // Number of shards
    std::vector<std::string> node_list; // List of nodes in the cluster
    std::string hash_function;    // "murmur", "fnv", etc.
    int replication_factor;       // Number of replicas per shard
    
    ShardingConfig() : num_shards(1), replication_factor(1) {}
};

/**
 * @brief Service for managing data sharding across distributed nodes
 * 
 * This service implements different sharding strategies (hash, range, vector-based)
 * to distribute vector data across multiple nodes in the cluster.
 */
class ShardingService {
public:
    enum class ShardingStrategy {
        HASH,
        RANGE,
        VECTOR,  // Based on vector content/similarity
        AUTO     // Automatic selection based on data characteristics
    };

private:
    std::shared_ptr<logging::Logger> logger_;
    ShardingConfig config_;
    std::vector<ShardInfo> shards_;
    std::unordered_map<std::string, std::vector<ShardInfo>> db_shards_; // database_id -> shards
    mutable std::mutex config_mutex_;
    
    // Hash function for hash-based sharding
    std::function<uint64_t(const std::string&)> hash_function_;
    
    // Range boundaries for range-based sharding
    std::vector<std::pair<std::string, std::string>> range_boundaries_;
    
public:
    explicit ShardingService();
    ~ShardingService() = default;
    
    // Initialize the sharding service with configuration
    bool initialize(const ShardingConfig& config);
    
    // Get the shard for a specific vector ID
    Result<std::string> get_shard_for_vector(const std::string& vector_id, 
                                           const std::string& database_id) const;
    
    // Get the node ID that hosts a specific shard
    Result<std::string> get_node_for_shard(const std::string& shard_id) const;
    
    // Get all shards for a specific database
    Result<std::vector<ShardInfo>> get_shards_for_database(const std::string& database_id) const;
    
    // Get the appropriate shard for a vector based on sharding strategy
    Result<ShardInfo> determine_shard(const Vector& vector, const Database& database) const;
    
    // Create shards for a new database
    Result<bool> create_shards_for_database(const Database& database);
    
    // Update sharding configuration
    Result<bool> update_sharding_config(const ShardingConfig& new_config);
    
    // Migrate data from one shard to another (for rebalancing)
    Result<bool> migrate_shard(const std::string& shard_id, const std::string& target_node_id);
    
    // Get current sharding configuration
    ShardingConfig get_config() const;
    
    // Get statistics about sharding distribution
    Result<std::unordered_map<std::string, size_t>> get_shard_distribution() const;
    
    // Check if sharding is balanced across nodes
    Result<bool> is_balanced() const;
    
    // Rebalance shards across nodes if needed
    Result<bool> rebalance_shards();
    
    // Handle node failure and redistribute its shards
    Result<bool> handle_node_failure(const std::string& failed_node_id);
    
    // Add a new node to the cluster and redistribute shards
    Result<bool> add_node_to_cluster(const std::string& node_id);
    
    // Remove a node from the cluster and redistribute its shards
    Result<bool> remove_node_from_cluster(const std::string& node_id);
    
    // Get the sharding strategy for a specific database
    ShardingStrategy get_strategy_for_database(const std::string& database_id) const;
    
    // Update shard metadata after operations
    Result<bool> update_shard_metadata(const std::string& shard_id, 
                                     size_t record_count, 
                                     size_t size_bytes);

private:
    // Initialize hash function based on configuration
    void initialize_hash_function();
    
    // Hash-based sharding implementation
    Result<ShardInfo> hash_based_sharding(const Vector& vector, const Database& database) const;
    
    // Range-based sharding implementation
    Result<ShardInfo> range_based_sharding(const Vector& vector, const Database& database) const;
    
    // Vector-based sharding implementation (based on vector content/similarity)
    Result<ShardInfo> vector_based_sharding(const Vector& vector, const Database& database) const;
    
    // Calculate shard number using hash function
    int calculate_shard_number(const std::string& key, int total_shards) const;
    
    // Generate shard ID
    std::string generate_shard_id(const std::string& database_id, int shard_number) const;
    
    // Validate sharding configuration
    bool validate_config(const ShardingConfig& config) const;
    
    // Create shards based on the selected strategy
    Result<bool> create_shards_by_strategy(const Database& database, ShardingStrategy strategy);
    
    // Distribute shards evenly across available nodes
    Result<bool> distribute_shards_to_nodes();
    
    // Check if a vector belongs to a specific range (for range-based sharding)
    bool vector_in_range(const Vector& vector, const std::pair<std::string, std::string>& range) const;
    
    // Update range boundaries for range-based sharding
    void update_range_boundaries();
};

} // namespace jadevectordb

#endif // JADEVECTORDB_SHARDING_SERVICE_H