#ifndef JADEVECTORDB_DATABASE_SERVICE_H
#define JADEVECTORDB_DATABASE_SERVICE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <shared_mutex>
#include <optional>

#include "models/database.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include "services/database_layer.h"
#include "services/cluster_service.h"
#include "services/sharding_service.h"

namespace jadevectordb {

// Database creation parameters structure
struct DatabaseCreationParams {
    std::string name;
    std::string description;
    int vectorDimension = 128;  // Default dimension
    std::string indexType = "HNSW";  // Default index type
    std::map<std::string, std::string> indexParameters;
    Database::ShardingConfig sharding = {"hash", 1};  // Default sharding
    Database::ReplicationConfig replication = {1, true};  // Default replication
    std::vector<Database::EmbeddingModel> embeddingModels;
    std::map<std::string, std::string> metadataSchema;
    std::unique_ptr<Database::RetentionPolicy> retentionPolicy;
    Database::AccessControl accessControl;
    
    // Validation
    bool validate() const {
        return !name.empty() && 
               vectorDimension > 0 && 
               vectorDimension <= 4096;  // Max dimension per spec
    }
};

// Database update parameters structure
struct DatabaseUpdateParams {
    std::optional<std::string> name;
    std::optional<std::string> description;
    std::optional<int> vectorDimension;
    std::optional<std::string> indexType;
    std::optional<std::unordered_map<std::string, std::string>> indexParameters;
    std::optional<Database::ShardingConfig> sharding;
    std::optional<Database::ReplicationConfig> replication;
    std::optional<std::vector<Database::EmbeddingModel>> embeddingModels;
    std::optional<std::unordered_map<std::string, std::string>> metadataSchema;
    std::optional<std::unique_ptr<Database::RetentionPolicy>> retentionPolicy;
    std::optional<Database::AccessControl> accessControl;
};

// Database listing parameters
struct DatabaseListParams {
    std::string filterByName;
    std::string filterByOwner;
    bool sortByName = false;
    int limit = 100;
    int offset = 0;
};

// Database service class
class DatabaseService {
public:
    // Database roles in distributed system
    enum class DatabaseRole {
        MASTER,      // Primary node responsible for database
        REPLICA,     // Replica node for redundancy
        OBSERVER     // Read-only observer node
    };

private:
    std::unique_ptr<DatabaseLayer> db_layer_;
    std::shared_ptr<logging::Logger> logger_;
    std::shared_ptr<ClusterService> cluster_service_;
    std::shared_ptr<ShardingService> sharding_service_;

public:
    explicit DatabaseService(
        std::unique_ptr<DatabaseLayer> db_layer = nullptr,
        std::shared_ptr<ClusterService> cluster_service = nullptr,
        std::shared_ptr<ShardingService> sharding_service = nullptr
    );
    ~DatabaseService() = default;
    
    // Initialize the service
    Result<void> initialize();
    
    // Initialize with distributed services
    Result<void> initialize_distributed(
        std::shared_ptr<ClusterService> cluster_service,
        std::shared_ptr<ShardingService> sharding_service
    );
    
    // Create a new database with distributed configuration
    Result<std::string> create_database(const DatabaseCreationParams& params);
    
    // Get database by ID (may route to appropriate node in distributed system)
    Result<Database> get_database(const std::string& database_id) const;
    
    // List databases with optional filtering
    Result<std::vector<Database>> list_databases(const DatabaseListParams& params = {}) const;
    
    // Update database configuration (distribute changes across cluster)
    Result<void> update_database(const std::string& database_id, const DatabaseUpdateParams& params);
    
    // Delete database (remove from all nodes in distributed system)
    Result<void> delete_database(const std::string& database_id);
    
    // Check if database exists (check across cluster in distributed system)
    Result<bool> database_exists(const std::string& database_id) const;
    
    // Get database role for the current node
    DatabaseRole get_role_for_database(const std::string& database_id) const;
    
    // Check if current node is master for database
    bool is_master_for_database(const std::string& database_id) const;
    
    // Get all database names
    Result<std::vector<std::string>> get_database_names() const;
    
    // Get database count
    Result<size_t> get_database_count() const;
    
    // Check database health
    Result<bool> check_database_health(const std::string& database_id) const;
    
    // Validate database creation parameters
    Result<void> validate_creation_params(const DatabaseCreationParams& params) const;
    
    // Validate database update parameters
    Result<void> validate_update_params(const DatabaseUpdateParams& params) const;
    
private:
    // Helper methods
    std::string generate_database_id() const;
    Database convert_params_to_database(const DatabaseCreationParams& params) const;
    void apply_update_params_to_database(Database& database, const DatabaseUpdateParams& params) const;
    
    // Create shards for the new database
    Result<void> create_database_shards(const std::string& database_id, const DatabaseCreationParams& params);
    
    // Distribute database configuration across cluster
    Result<void> distribute_config_to_nodes(const Database& database);
    
    // Validate cluster state before distributed operations
    Result<void> validate_cluster_state() const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_DATABASE_SERVICE_H