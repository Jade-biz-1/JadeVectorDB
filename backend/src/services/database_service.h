#ifndef JADEVECTORDB_DATABASE_SERVICE_H
#define JADEVECTORDB_DATABASE_SERVICE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <shared_mutex>

#include "models/database.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include "services/database_layer.h"

namespace jadevectordb {

// Database creation parameters structure
struct DatabaseCreationParams {
    std::string name;
    std::string description;
    int vectorDimension = 128;  // Default dimension
    std::string indexType = "HNSW";  // Default index type
    std::unordered_map<std::string, std::string> indexParameters;
    Database::ShardingConfig sharding = {"hash", 1};  // Default sharding
    Database::ReplicationConfig replication = {1, true};  // Default replication
    std::vector<Database::EmbeddingModel> embeddingModels;
    std::unordered_map<std::string, std::string> metadataSchema;
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
    int limit = 100;
    int offset = 0;
};

// Database service class
class DatabaseService {
private:
    std::unique_ptr<DatabaseLayer> db_layer_;
    std::shared_ptr<logging::Logger> logger_;

public:
    explicit DatabaseService(std::unique_ptr<DatabaseLayer> db_layer = nullptr);
    ~DatabaseService() = default;
    
    // Initialize the service
    Result<void> initialize();
    
    // Create a new database
    Result<std::string> create_database(const DatabaseCreationParams& params);
    
    // Get database by ID
    Result<Database> get_database(const std::string& database_id) const;
    
    // List databases with optional filtering
    Result<std::vector<Database>> list_databases(const DatabaseListParams& params = {}) const;
    
    // Update database configuration
    Result<void> update_database(const std::string& database_id, const DatabaseUpdateParams& params);
    
    // Delete database
    Result<void> delete_database(const std::string& database_id);
    
    // Check if database exists
    Result<bool> database_exists(const std::string& database_id) const;
    
    // Get database count
    Result<size_t> get_database_count() const;
    
    // Validate database creation parameters
    Result<void> validate_creation_params(const DatabaseCreationParams& params) const;
    
    // Validate database update parameters
    Result<void> validate_update_params(const DatabaseUpdateParams& params) const;
    
    // Get database statistics
    Result<std::unordered_map<std::string, std::string>> get_database_stats(const std::string& database_id) const;

private:
    // Helper methods
    std::string generate_database_id() const;
    Database convert_params_to_database(const DatabaseCreationParams& params) const;
    void apply_update_params_to_database(Database& database, const DatabaseUpdateParams& params) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_DATABASE_SERVICE_H