#ifndef JADEVECTORDB_VECTOR_STORAGE_SERVICE_H
#define JADEVECTORDB_VECTOR_STORAGE_SERVICE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <shared_mutex>

#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"
#include "lib/logging.h"
#include "services/database_layer.h"
#include "services/sharding_service.h"
#include "services/query_router.h"
#include "services/replication_service.h"

namespace jadevectordb {

class VectorStorageService {
private:
    std::unique_ptr<DatabaseLayer> db_layer_;
    std::shared_ptr<logging::Logger> logger_;
    std::shared_ptr<ShardingService> sharding_service_;
    std::shared_ptr<QueryRouter> query_router_;
    std::shared_ptr<ReplicationService> replication_service_;

public:
    explicit VectorStorageService(
        std::unique_ptr<DatabaseLayer> db_layer = nullptr,
        std::shared_ptr<ShardingService> sharding_service = nullptr,
        std::shared_ptr<QueryRouter> query_router = nullptr,
        std::shared_ptr<ReplicationService> replication_service = nullptr
    );
    ~VectorStorageService() = default;

    // Initialize the service
    Result<void> initialize();

    // Store a single vector (distributes to appropriate shard(s))
    Result<void> store_vector(const std::string& database_id, const Vector& vector);

    // Store multiple vectors in a batch (distributes across shards)
    Result<void> batch_store_vectors(const std::string& database_id, 
                                   const std::vector<Vector>& vectors);

    // Retrieve a single vector by ID (routes to appropriate shard)
    Result<Vector> retrieve_vector(const std::string& database_id, const std::string& vector_id) const;

    // Retrieve multiple vectors by ID (routes to appropriate shards)
    Result<std::vector<Vector>> retrieve_vectors(const std::string& database_id, 
                                               const std::vector<std::string>& vector_ids) const;

    // Update an existing vector (distributes to appropriate shard(s))
    Result<void> update_vector(const std::string& database_id, const Vector& vector);

    // Delete a vector (routes to appropriate shard)
    Result<void> delete_vector(const std::string& database_id, const std::string& vector_id);

    // Delete multiple vectors
    Result<void> batch_delete_vectors(const std::string& database_id,
                                    const std::vector<std::string>& vector_ids);

    // Check if a vector exists (routes to appropriate shard)
    Result<bool> vector_exists(const std::string& database_id, const std::string& vector_id) const;

    // Get the number of vectors in a database
    Result<size_t> get_vector_count(const std::string& database_id) const;

    // Validate a vector before storing
    Result<void> validate_vector(const std::string& database_id, const Vector& vector) const;

    // Get all vector IDs in a database
    Result<std::vector<std::string>> get_all_vector_ids(const std::string& database_id) const;
    
    // Initialize with distributed services
    Result<void> initialize_distributed(
        std::shared_ptr<ShardingService> sharding_service,
        std::shared_ptr<QueryRouter> query_router,
        std::shared_ptr<ReplicationService> replication_service
    );
    
    // Replicate vector to appropriate nodes according to replication policy
    Result<void> replicate_vector(const Vector& vector, const std::string& database_id);
    
    // Handle vector migration when sharding changes
    Result<void> migrate_vector(const std::string& vector_id, 
                              const std::string& source_shard,
                              const std::string& target_shard);

private:
    // Helper to get the appropriate shard for a vector
    Result<std::string> get_target_shard(const std::string& vector_id, 
                                       const std::string& database_id) const;
    
    // Internal method to store vector on a specific shard
    Result<void> store_vector_on_shard(const std::string& shard_id, const Vector& vector);
    
    // Internal method to retrieve vector from a specific shard
    Result<Vector> retrieve_vector_from_shard(const std::string& shard_id, 
                                            const std::string& vector_id) const;
    
    // Check cluster health before storage operations
    Result<bool> check_cluster_health() const;

    // Validate cluster state before distributed operations
    Result<void> validate_cluster_state() const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_VECTOR_STORAGE_SERVICE_H