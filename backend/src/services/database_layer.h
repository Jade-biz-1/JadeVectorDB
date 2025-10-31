#ifndef JADEVECTORDB_DATABASE_LAYER_H
#define JADEVECTORDB_DATABASE_LAYER_H

#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "models/database.h"
#include "models/vector.h"
#include "models/index.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

// Forward declarations for distributed services
namespace jadevectordb {
    class ShardingService;
    class ReplicationService;
    class QueryRouter;
}

namespace jadevectordb {

// Forward declaration
class DatabaseLayer;

// Interface for database persistence operations
class DatabasePersistenceInterface {
public:
    virtual ~DatabasePersistenceInterface() = default;
    
    // Database operations
    virtual Result<std::string> create_database(const Database& db) = 0;
    virtual Result<Database> get_database(const std::string& database_id) = 0;
    virtual Result<std::vector<Database>> list_databases() = 0;
    virtual Result<void> update_database(const std::string& database_id, const Database& db) = 0;
    virtual Result<void> delete_database(const std::string& database_id) = 0;
    
    // Vector operations
    virtual Result<void> store_vector(const std::string& database_id, const Vector& vector) = 0;
    virtual Result<Vector> retrieve_vector(const std::string& database_id, const std::string& vector_id) = 0;
    virtual Result<std::vector<Vector>> retrieve_vectors(const std::string& database_id, 
                                                        const std::vector<std::string>& vector_ids) = 0;
    virtual Result<void> update_vector(const std::string& database_id, const Vector& vector) = 0;
    virtual Result<void> delete_vector(const std::string& database_id, const std::string& vector_id) = 0;
    
    // Batch operations
    virtual Result<void> batch_store_vectors(const std::string& database_id, 
                                           const std::vector<Vector>& vectors) = 0;
    virtual Result<void> batch_delete_vectors(const std::string& database_id,
                                            const std::vector<std::string>& vector_ids) = 0;
    
    // Index operations
    virtual Result<void> create_index(const std::string& database_id, const Index& index) = 0;
    virtual Result<Index> get_index(const std::string& database_id, const std::string& index_id) = 0;
    virtual Result<std::vector<Index>> list_indexes(const std::string& database_id) = 0;
    virtual Result<void> update_index(const std::string& database_id, const std::string& index_id, const Index& index) = 0;
    virtual Result<void> delete_index(const std::string& database_id, const std::string& index_id) = 0;
    
    // Validation
    virtual Result<bool> database_exists(const std::string& database_id) const = 0;
    virtual Result<bool> vector_exists(const std::string& database_id, const std::string& vector_id) const = 0;
    virtual Result<bool> index_exists(const std::string& database_id, const std::string& index_id) const = 0;
    
    // Utility methods
    virtual Result<size_t> get_vector_count(const std::string& database_id) const = 0;
    virtual Result<std::vector<std::string>> get_all_vector_ids(const std::string& database_id) const = 0;
};

// In-memory implementation of the database persistence interface
class InMemoryDatabasePersistence : public DatabasePersistenceInterface {
private:
    std::unordered_map<std::string, Database> databases_;
    std::unordered_map<std::string, std::unordered_map<std::string, Vector>> vectors_by_db_;
    std::unordered_map<std::string, std::unordered_map<std::string, Index>> indexes_by_db_;
    
    mutable std::shared_mutex databases_mutex_;
    mutable std::shared_mutex vectors_mutex_;
    mutable std::shared_mutex indexes_mutex_;
    
    std::shared_ptr<logging::Logger> logger_;
    
    // Distributed services
    std::shared_ptr<ShardingService> sharding_service_;
    std::shared_ptr<QueryRouter> query_router_;
    std::shared_ptr<ReplicationService> replication_service_;

public:
    explicit InMemoryDatabasePersistence(
        std::shared_ptr<ShardingService> sharding_service = nullptr,
        std::shared_ptr<QueryRouter> query_router = nullptr,
        std::shared_ptr<ReplicationService> replication_service = nullptr
    );
    ~InMemoryDatabasePersistence() = default;
    
    // Database operations
    Result<std::string> create_database(const Database& db) override;
    Result<Database> get_database(const std::string& database_id) override;
    Result<std::vector<Database>> list_databases() override;
    Result<void> update_database(const std::string& database_id, const Database& db) override;
    Result<void> delete_database(const std::string& database_id) override;
    
    // Vector operations
    Result<void> store_vector(const std::string& database_id, const Vector& vector) override;
    Result<Vector> retrieve_vector(const std::string& database_id, const std::string& vector_id) override;
    Result<std::vector<Vector>> retrieve_vectors(const std::string& database_id, 
                                               const std::vector<std::string>& vector_ids) override;
    Result<void> update_vector(const std::string& database_id, const Vector& vector) override;
    Result<void> delete_vector(const std::string& database_id, const std::string& vector_id) override;
    
    // Batch operations
    Result<void> batch_store_vectors(const std::string& database_id, 
                                   const std::vector<Vector>& vectors) override;
    Result<void> batch_delete_vectors(const std::string& database_id,
                                    const std::vector<std::string>& vector_ids) override;
    
    // Index operations
    Result<void> create_index(const std::string& database_id, const Index& index) override;
    Result<Index> get_index(const std::string& database_id, const std::string& index_id) override;
    Result<std::vector<Index>> list_indexes(const std::string& database_id) override;
    Result<void> update_index(const std::string& database_id, const std::string& index_id, const Index& index) override;
    Result<void> delete_index(const std::string& database_id, const std::string& index_id) override;
    
    // Validation
    Result<bool> database_exists(const std::string& database_id) const override;
    Result<bool> vector_exists(const std::string& database_id, const std::string& vector_id) const override;
    Result<bool> index_exists(const std::string& database_id, const std::string& index_id) const override;
    
    // Utility methods
    Result<size_t> get_vector_count(const std::string& database_id) const override;
    Result<std::vector<std::string>> get_all_vector_ids(const std::string& database_id) const override;
    
private:
    std::string generate_id() const;
    bool validate_vector_dimensions(const Vector& vector, int expected_dimension) const;
};

// Main database abstraction layer
class DatabaseLayer {
private:
    std::unique_ptr<DatabasePersistenceInterface> persistence_layer_;
    std::shared_ptr<logging::Logger> logger_;
    
    // Distributed services
    std::shared_ptr<ShardingService> sharding_service_;
    std::shared_ptr<QueryRouter> query_router_;
    std::shared_ptr<ReplicationService> replication_service_;

public:
    explicit DatabaseLayer(
        std::unique_ptr<DatabasePersistenceInterface> persistence = nullptr,
        std::shared_ptr<ShardingService> sharding_service = nullptr,
        std::shared_ptr<QueryRouter> query_router = nullptr,
        std::shared_ptr<ReplicationService> replication_service = nullptr
    );
    ~DatabaseLayer() = default;
    
    // Initialize the database layer with a specific persistence implementation
    Result<void> initialize();
    
    // Database operations
    Result<std::string> create_database(const Database& db_config);
    Result<Database> get_database(const std::string& database_id) const;
    Result<std::vector<Database>> list_databases() const;
    Result<void> update_database(const std::string& database_id, const Database& new_config);
    Result<void> delete_database(const std::string& database_id);
    
    // Vector operations
    Result<void> store_vector(const std::string& database_id, const Vector& vector);
    Result<Vector> retrieve_vector(const std::string& database_id, const std::string& vector_id) const;
    Result<std::vector<Vector>> retrieve_vectors(const std::string& database_id, 
                                                const std::vector<std::string>& vector_ids) const;
    Result<void> update_vector(const std::string& database_id, const Vector& vector);
    Result<void> delete_vector(const std::string& database_id, const std::string& vector_id);
    
    // Batch operations
    Result<void> batch_store_vectors(const std::string& database_id, 
                                   const std::vector<Vector>& vectors);
    Result<void> batch_delete_vectors(const std::string& database_id,
                                    const std::vector<std::string>& vector_ids);
    
    // Index operations
    Result<void> create_index(const std::string& database_id, const Index& index);
    Result<Index> get_index(const std::string& database_id, const std::string& index_id) const;
    Result<std::vector<Index>> list_indexes(const std::string& database_id) const;
    Result<void> update_index(const std::string& database_id, const std::string& index_id, const Index& index);
    Result<void> delete_index(const std::string& database_id, const std::string& index_id);
    
    // Validation
    Result<bool> database_exists(const std::string& database_id) const;
    Result<bool> vector_exists(const std::string& database_id, const std::string& vector_id) const;
    Result<bool> index_exists(const std::string& database_id, const std::string& index_id) const;
    
    // Utility methods
    Result<size_t> get_database_count() const;
    Result<size_t> get_vector_count(const std::string& database_id) const;
    Result<size_t> get_index_count(const std::string& database_id) const;
    
    // Additional utility methods
    Result<std::vector<std::string>> get_all_vector_ids(const std::string& database_id) const;
    
private:
    // Helper method for vector replication
    Result<void> replicate_vector(const Vector& vector, const std::string& database_id);
};

} // namespace jadevectordb

#endif // JADEVECTORDB_DATABASE_LAYER_H