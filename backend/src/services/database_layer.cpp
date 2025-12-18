#include "database_layer.h"
#include "lib/logging.h"
#include "services/sharding_service.h"
#include "services/replication_service.h"
#include "services/query_router.h"
#include "storage/memory_mapped_vector_store.h"
#include <random>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <fstream>

namespace jadevectordb {

// InMemoryDatabasePersistence implementation
InMemoryDatabasePersistence::InMemoryDatabasePersistence(
    std::shared_ptr<ShardingService> sharding_service,
    std::shared_ptr<QueryRouter> query_router,
    std::shared_ptr<ReplicationService> replication_service)
    : sharding_service_(sharding_service)
    , query_router_(query_router)
    , replication_service_(replication_service) {
    logger_ = logging::LoggerManager::get_logger("InMemoryDatabasePersistence");
}

Result<std::string> InMemoryDatabasePersistence::create_database(const Database& db) {
    std::unique_lock<std::shared_mutex> lock(databases_mutex_);
    
    if (!db.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid database configuration");
    }
    
    // Use the database ID from the input if provided, otherwise generate a new one
    std::string database_id = db.databaseId.empty() ? generate_id() : db.databaseId;
    Database new_db = db;
    new_db.databaseId = database_id;
    new_db.created_at = "2025-10-11T00:00:00Z"; // In a real system, use current timestamp
    new_db.updated_at = "2025-10-11T00:00:00Z";
    
    databases_[database_id] = new_db;
    
    // Initialize the vectors and indexes maps for this database
    vectors_by_db_[database_id] = std::unordered_map<std::string, Vector>();
    indexes_by_db_[database_id] = std::unordered_map<std::string, Index>();
    
    LOG_INFO(logger_, "Created database: " << database_id);
    return database_id;
}

Result<Database> InMemoryDatabasePersistence::get_database(const std::string& database_id) {
    std::shared_lock<std::shared_mutex> lock(databases_mutex_);
    
    auto it = databases_.find(database_id);
    if (it == databases_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    return it->second;
}

Result<std::vector<Database>> InMemoryDatabasePersistence::list_databases() {
    std::shared_lock<std::shared_mutex> lock(databases_mutex_);
    
    std::vector<Database> result;
    for (const auto& [id, db] : databases_) {
        result.push_back(db);
    }
    
    return result;
}

Result<void> InMemoryDatabasePersistence::update_database(const std::string& database_id, const Database& db) {
    std::unique_lock<std::shared_mutex> lock(databases_mutex_);
    
    if (!db.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid database configuration");
    }
    
    auto it = databases_.find(database_id);
    if (it == databases_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    Database updated_db = db;
    updated_db.databaseId = database_id;
    updated_db.updated_at = "2025-10-11T00:00:00Z"; // In a real system, use current timestamp
    
    databases_[database_id] = updated_db;
    
    LOG_INFO(logger_, "Updated database: " << database_id);
    return {};
}

Result<void> InMemoryDatabasePersistence::delete_database(const std::string& database_id) {
    std::unique_lock<std::shared_mutex> lock(databases_mutex_);
    
    auto it = databases_.find(database_id);
    if (it == databases_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    databases_.erase(it);
    vectors_by_db_.erase(database_id);
    indexes_by_db_.erase(database_id);
    
    LOG_INFO(logger_, "Deleted database: " << database_id);
    return {};
}

Result<void> InMemoryDatabasePersistence::store_vector(const std::string& database_id, const Vector& vector) {
    // First check if database exists
    if (!database_exists(database_id).value_or(false)) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database does not exist: " + database_id);
    }
    
    {
        std::shared_lock<std::shared_mutex> db_lock(databases_mutex_);
        auto db_it = databases_.find(database_id);
        if (db_it == databases_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
        }
        
        // Validate vector dimensions match database configuration
        if (!validate_vector_dimensions(vector, db_it->second.vectorDimension)) {
            RETURN_ERROR(ErrorCode::VECTOR_DIMENSION_MISMATCH, 
                        "Vector dimension mismatch. Expected: " + 
                        std::to_string(db_it->second.vectorDimension) + 
                        ", got: " + std::to_string(vector.values.size()));
        }
    }
    
    std::unique_lock<std::shared_mutex> vectors_lock(vectors_mutex_);
    vectors_by_db_[database_id][vector.id] = vector;
    
    LOG_DEBUG(logger_, "Stored vector: " << vector.id << " in database: " << database_id);
    return {};
}

Result<Vector> InMemoryDatabasePersistence::retrieve_vector(const std::string& database_id, const std::string& vector_id) {
    std::shared_lock<std::shared_mutex> lock(vectors_mutex_);
    
    auto db_it = vectors_by_db_.find(database_id);
    if (db_it == vectors_by_db_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    auto vector_it = db_it->second.find(vector_id);
    if (vector_it == db_it->second.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Vector not found: " + vector_id + " in database: " + database_id);
    }
    
    return vector_it->second;
}

Result<std::vector<Vector>> InMemoryDatabasePersistence::retrieve_vectors(
    const std::string& database_id, 
    const std::vector<std::string>& vector_ids) {
    
    std::shared_lock<std::shared_mutex> lock(vectors_mutex_);
    
    auto db_it = vectors_by_db_.find(database_id);
    if (db_it == vectors_by_db_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    std::vector<Vector> result;
    for (const auto& vector_id : vector_ids) {
        auto vector_it = db_it->second.find(vector_id);
        if (vector_it != db_it->second.end()) {
            result.push_back(vector_it->second);
        }
    }
    
    return result;
}

Result<void> InMemoryDatabasePersistence::update_vector(const std::string& database_id, const Vector& vector) {
    // First check if database exists
    if (!database_exists(database_id).value_or(false)) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database does not exist: " + database_id);
    }
    
    {
        std::shared_lock<std::shared_mutex> db_lock(databases_mutex_);
        auto db_it = databases_.find(database_id);
        if (db_it == databases_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
        }
        
        // Validate vector dimensions match database configuration
        if (!validate_vector_dimensions(vector, db_it->second.vectorDimension)) {
            RETURN_ERROR(ErrorCode::VECTOR_DIMENSION_MISMATCH, 
                        "Vector dimension mismatch. Expected: " + 
                        std::to_string(db_it->second.vectorDimension) + 
                        ", got: " + std::to_string(vector.values.size()));
        }
    }
    
    std::unique_lock<std::shared_mutex> vectors_lock(vectors_mutex_);
    
    auto db_vectors_it = vectors_by_db_.find(database_id);
    if (db_vectors_it == vectors_by_db_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    auto vector_it = db_vectors_it->second.find(vector.id);
    if (vector_it == db_vectors_it->second.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Vector not found: " + vector.id + " in database: " + database_id);
    }
    
    // Update the vector
    vector_it->second = vector;
    
    LOG_DEBUG(logger_, "Updated vector: " << vector.id << " in database: " << database_id);
    return {};
}

Result<void> InMemoryDatabasePersistence::delete_vector(const std::string& database_id, const std::string& vector_id) {
    std::unique_lock<std::shared_mutex> vectors_lock(vectors_mutex_);
    
    auto db_it = vectors_by_db_.find(database_id);
    if (db_it == vectors_by_db_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    auto vector_it = db_it->second.find(vector_id);
    if (vector_it == db_it->second.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Vector not found: " + vector_id + " in database: " + database_id);
    }
    
    db_it->second.erase(vector_it);
    
    LOG_DEBUG(logger_, "Deleted vector: " << vector_id << " from database: " << database_id);
    return {};
}

Result<void> InMemoryDatabasePersistence::batch_store_vectors(
    const std::string& database_id, 
    const std::vector<Vector>& vectors) {
    
    // First check if database exists
    if (!database_exists(database_id).value_or(false)) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database does not exist: " + database_id);
    }
    
    {
        std::shared_lock<std::shared_mutex> db_lock(databases_mutex_);
        auto db_it = databases_.find(database_id);
        if (db_it == databases_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
        }
        
        // Validate vector dimensions match database configuration
        for (const auto& vector : vectors) {
            if (!validate_vector_dimensions(vector, db_it->second.vectorDimension)) {
                RETURN_ERROR(ErrorCode::VECTOR_DIMENSION_MISMATCH, 
                            "Vector dimension mismatch for vector: " + vector.id + 
                            ". Expected: " + std::to_string(db_it->second.vectorDimension) + 
                            ", got: " + std::to_string(vector.values.size()));
            }
        }
    }
    
    std::unique_lock<std::shared_mutex> vectors_lock(vectors_mutex_);
    
    for (const auto& vector : vectors) {
        vectors_by_db_[database_id][vector.id] = vector;
    }
    
    LOG_DEBUG(logger_, "Batch stored " << vectors.size() << " vectors in database: " << database_id);
    return {};
}

Result<void> InMemoryDatabasePersistence::batch_delete_vectors(
    const std::string& database_id,
    const std::vector<std::string>& vector_ids) {
    
    std::unique_lock<std::shared_mutex> vectors_lock(vectors_mutex_);
    
    auto db_it = vectors_by_db_.find(database_id);
    if (db_it == vectors_by_db_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    for (const auto& vector_id : vector_ids) {
        db_it->second.erase(vector_id);
    }
    
    LOG_DEBUG(logger_, "Batch deleted " << vector_ids.size() << " vectors from database: " << database_id);
    return {};
}

Result<void> InMemoryDatabasePersistence::create_index(const std::string& database_id, const Index& index) {
    // First check if database exists
    if (!database_exists(database_id).value_or(false)) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database does not exist: " + database_id);
    }
    
    std::unique_lock<std::shared_mutex> indexes_lock(indexes_mutex_);
    
    std::string index_id = generate_id();
    Index new_index = index;
    new_index.indexId = index_id;
    new_index.databaseId = database_id;
    new_index.created_at = "2025-10-11T00:00:00Z";
    new_index.updated_at = "2025-10-11T00:00:00Z";
    
    indexes_by_db_[database_id][index_id] = new_index;
    
    LOG_INFO(logger_, "Created index: " << index_id << " in database: " << database_id);
    return {};
}

Result<Index> InMemoryDatabasePersistence::get_index(const std::string& database_id, const std::string& index_id) {
    std::shared_lock<std::shared_mutex> lock(indexes_mutex_);
    
    auto db_it = indexes_by_db_.find(database_id);
    if (db_it == indexes_by_db_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    auto index_it = db_it->second.find(index_id);
    if (index_it == db_it->second.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Index not found: " + index_id + " in database: " + database_id);
    }
    
    return index_it->second;
}

Result<std::vector<Index>> InMemoryDatabasePersistence::list_indexes(const std::string& database_id) {
    std::shared_lock<std::shared_mutex> lock(indexes_mutex_);
    
    auto db_it = indexes_by_db_.find(database_id);
    if (db_it == indexes_by_db_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    std::vector<Index> result;
    for (const auto& [id, index] : db_it->second) {
        result.push_back(index);
    }
    
    return result;
}

Result<void> InMemoryDatabasePersistence::update_index(const std::string& database_id, 
                                                      const std::string& index_id, 
                                                      const Index& index) {
    std::unique_lock<std::shared_mutex> indexes_lock(indexes_mutex_);
    
    auto db_it = indexes_by_db_.find(database_id);
    if (db_it == indexes_by_db_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    auto index_it = db_it->second.find(index_id);
    if (index_it == db_it->second.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Index not found: " + index_id + " in database: " + database_id);
    }
    
    Index updated_index = index;
    updated_index.indexId = index_id;
    updated_index.databaseId = database_id;
    updated_index.updated_at = "2025-10-11T00:00:00Z";
    
    index_it->second = updated_index;
    
    LOG_INFO(logger_, "Updated index: " << index_id << " in database: " << database_id);
    return {};
}

Result<void> InMemoryDatabasePersistence::delete_index(const std::string& database_id, const std::string& index_id) {
    std::unique_lock<std::shared_mutex> indexes_lock(indexes_mutex_);
    
    auto db_it = indexes_by_db_.find(database_id);
    if (db_it == indexes_by_db_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    auto index_it = db_it->second.find(index_id);
    if (index_it == db_it->second.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Index not found: " + index_id + " in database: " + database_id);
    }
    
    db_it->second.erase(index_it);
    
    LOG_INFO(logger_, "Deleted index: " << index_id << " from database: " << database_id);
    return {};
}

Result<bool> InMemoryDatabasePersistence::database_exists(const std::string& database_id) const {
    std::shared_lock<std::shared_mutex> lock(databases_mutex_);
    return databases_.find(database_id) != databases_.end();
}

Result<bool> InMemoryDatabasePersistence::vector_exists(const std::string& database_id, const std::string& vector_id) const {
    std::shared_lock<std::shared_mutex> lock(vectors_mutex_);
    
    auto db_it = vectors_by_db_.find(database_id);
    if (db_it == vectors_by_db_.end()) {
        return false;
    }
    
    return db_it->second.find(vector_id) != db_it->second.end();
}

Result<bool> InMemoryDatabasePersistence::index_exists(const std::string& database_id, const std::string& index_id) const {
    std::shared_lock<std::shared_mutex> lock(indexes_mutex_);
    
    auto db_it = indexes_by_db_.find(database_id);
    if (db_it == indexes_by_db_.end()) {
        return false;
    }
    
    return db_it->second.find(index_id) != db_it->second.end();
}

Result<size_t> InMemoryDatabasePersistence::get_vector_count(const std::string& database_id) const {
    std::shared_lock<std::shared_mutex> lock(vectors_mutex_);
    
    auto db_it = vectors_by_db_.find(database_id);
    if (db_it == vectors_by_db_.end()) {
        return 0; // If database doesn't exist, return 0
    }
    
    return db_it->second.size();
}

Result<std::vector<std::string>> InMemoryDatabasePersistence::get_all_vector_ids(const std::string& database_id) const {
    std::shared_lock<std::shared_mutex> lock(vectors_mutex_);
    
    auto db_it = vectors_by_db_.find(database_id);
    if (db_it == vectors_by_db_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    std::vector<std::string> vector_ids;
    for (const auto& [id, vector] : db_it->second) {
        vector_ids.push_back(id);
    }
    
    return vector_ids;
}

std::string InMemoryDatabasePersistence::generate_id() const {
    // Generate a unique ID
    // In a real implementation, this should use a more robust ID generation method
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto count = duration.count();
    
    std::stringstream ss;
    ss << std::hex << count;
    return ss.str();
}

bool InMemoryDatabasePersistence::validate_vector_dimensions(const Vector& vector, int expected_dimension) const {
    return vector.values.size() == static_cast<size_t>(expected_dimension);
}

// DatabaseLayer implementation
DatabaseLayer::DatabaseLayer(
    std::unique_ptr<DatabasePersistenceInterface> persistence,
    std::shared_ptr<ShardingService> sharding_service,
    std::shared_ptr<QueryRouter> query_router,
    std::shared_ptr<ReplicationService> replication_service)
    : sharding_service_(sharding_service)
    , query_router_(query_router)
    , replication_service_(replication_service)
    , persistence_layer_(std::move(persistence)) {
    
    if (!persistence_layer_) {
        // If no persistence layer is provided, use in-memory implementation
        // Pass distributed services to the persistence layer
        persistence_layer_ = std::make_unique<InMemoryDatabasePersistence>(
            sharding_service_,
            query_router_,
            replication_service_
        );
    }
    
    logger_ = logging::LoggerManager::get_logger("DatabaseLayer");
}

Result<void> DatabaseLayer::initialize() {
    LOG_INFO(logger_, "Database layer initialized");
    return {};
}

Result<std::string> DatabaseLayer::create_database(const Database& db_config) {
    if (!db_config.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid database configuration");
    }
    
    auto result = persistence_layer_->create_database(db_config);
    if (result.has_value()) {
        LOG_INFO(logger_, "Created database with ID: " << result.value());
    }
    
    return result;
}

Result<Database> DatabaseLayer::get_database(const std::string& database_id) const {
    return persistence_layer_->get_database(database_id);
}

Result<std::vector<Database>> DatabaseLayer::list_databases() const {
    return persistence_layer_->list_databases();
}

Result<void> DatabaseLayer::update_database(const std::string& database_id, const Database& new_config) {
    if (!new_config.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid database configuration");
    }
    
    return persistence_layer_->update_database(database_id, new_config);
}

Result<void> DatabaseLayer::delete_database(const std::string& database_id) {
    return persistence_layer_->delete_database(database_id);
}

Result<void> DatabaseLayer::store_vector(const std::string& database_id, const Vector& vector) {
    if (!vector.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid vector data");
    }
    
    // For distributed systems, determine the target shard
    std::string target_shard = database_id; // Default to database ID as shard if no sharding service
    if (sharding_service_) {
        auto shard_result = sharding_service_->get_shard_for_vector(vector.id, database_id);
        if (shard_result.has_value()) {
            target_shard = shard_result.value();
        } else {
            LOG_WARN(logger_, "Could not determine shard for vector " << vector.id << 
                    ", using default: " << target_shard);
        }
    }
    
    // Store vector in the determined shard
    auto result = persistence_layer_->store_vector(target_shard, vector);
    
    if (result.has_value()) {
        LOG_DEBUG(logger_, "Stored vector: " << vector.id << " in database: " << database_id << 
                 " on shard: " << target_shard);
        
        // Replicate the vector to other nodes if replication is enabled
        if (replication_service_) {
            auto replication_result = replicate_vector(vector, database_id);
            if (!replication_result.has_value()) {
                LOG_WARN(logger_, "Replication failed for vector " << vector.id << 
                        ": " << ErrorHandler::format_error(replication_result.error()));
            }
        }
    }
    
    return result;
}

Result<Vector> DatabaseLayer::retrieve_vector(const std::string& database_id, const std::string& vector_id) const {
    // For distributed systems, determine the source shard
    std::string source_shard = database_id; // Default to database ID as shard if no sharding service
    if (sharding_service_) {
        auto shard_result = sharding_service_->get_shard_for_vector(vector_id, database_id);
        if (shard_result.has_value()) {
            source_shard = shard_result.value();
        } else {
            LOG_WARN(logger_, "Could not determine shard for vector " << vector_id << 
                    ", using default: " << source_shard);
        }
    }
    
    return persistence_layer_->retrieve_vector(source_shard, vector_id);
}

Result<std::vector<Vector>> DatabaseLayer::retrieve_vectors(const std::string& database_id, 
                                                          const std::vector<std::string>& vector_ids) const {
    return persistence_layer_->retrieve_vectors(database_id, vector_ids);
}

Result<void> DatabaseLayer::update_vector(const std::string& database_id, const Vector& vector) {
    if (!vector.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid vector data");
    }
    
    return persistence_layer_->update_vector(database_id, vector);
}

Result<void> DatabaseLayer::delete_vector(const std::string& database_id, const std::string& vector_id) {
    return persistence_layer_->delete_vector(database_id, vector_id);
}

Result<void> DatabaseLayer::batch_store_vectors(const std::string& database_id, 
                                              const std::vector<Vector>& vectors) {
    for (const auto& vector : vectors) {
        if (!vector.validate()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid vector data in batch: " + vector.id);
        }
    }
    
    return persistence_layer_->batch_store_vectors(database_id, vectors);
}

Result<void> DatabaseLayer::batch_delete_vectors(const std::string& database_id,
                                               const std::vector<std::string>& vector_ids) {
    return persistence_layer_->batch_delete_vectors(database_id, vector_ids);
}

Result<void> DatabaseLayer::create_index(const std::string& database_id, const Index& index) {
    if (!index.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid index configuration");
    }
    
    return persistence_layer_->create_index(database_id, index);
}

Result<Index> DatabaseLayer::get_index(const std::string& database_id, const std::string& index_id) const {
    return persistence_layer_->get_index(database_id, index_id);
}

Result<std::vector<Index>> DatabaseLayer::list_indexes(const std::string& database_id) const {
    return persistence_layer_->list_indexes(database_id);
}

Result<void> DatabaseLayer::update_index(const std::string& database_id, const std::string& index_id, const Index& index) {
    if (!index.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid index configuration");
    }
    
    return persistence_layer_->update_index(database_id, index_id, index);
}

Result<void> DatabaseLayer::delete_index(const std::string& database_id, const std::string& index_id) {
    return persistence_layer_->delete_index(database_id, index_id);
}

Result<bool> DatabaseLayer::database_exists(const std::string& database_id) const {
    return persistence_layer_->database_exists(database_id);
}

Result<bool> DatabaseLayer::vector_exists(const std::string& database_id, const std::string& vector_id) const {
    return persistence_layer_->vector_exists(database_id, vector_id);
}

Result<bool> DatabaseLayer::index_exists(const std::string& database_id, const std::string& index_id) const {
    return persistence_layer_->index_exists(database_id, index_id);
}

Result<size_t> DatabaseLayer::get_database_count() const {
    auto result = persistence_layer_->list_databases();
    if (!result.has_value()) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to get database count");
    }
    
    return result.value().size();
}

Result<size_t> DatabaseLayer::get_vector_count(const std::string& database_id) const {
    return persistence_layer_->get_vector_count(database_id);
}

Result<size_t> DatabaseLayer::get_index_count(const std::string& database_id) const {
    auto result = persistence_layer_->list_indexes(database_id);
    if (!result.has_value()) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to get index count");
    }
    
    return result.value().size();
}

Result<std::vector<std::string>> DatabaseLayer::get_all_vector_ids(const std::string& database_id) const {
    return persistence_layer_->get_all_vector_ids(database_id);
}

Result<void> DatabaseLayer::replicate_vector(const Vector& vector, const std::string& database_id) {
    if (!replication_service_) {
        RETURN_ERROR(ErrorCode::SERVICE_UNAVAILABLE, "Replication service not available");
    }
    
    Database dummy_db;
    dummy_db.databaseId = database_id;
    
    auto result = replication_service_->replicate_vector(vector, dummy_db);
    if (!result.has_value()) {
        LOG_ERROR(logger_, "Failed to replicate vector " << vector.id << 
                 ": " << ErrorHandler::format_error(result.error()));
        return result;
    }
    
    LOG_DEBUG(logger_, "Successfully replicated vector: " << vector.id);
    return {};
}

// ============================================================================
// PersistentDatabasePersistence implementation
// ============================================================================

PersistentDatabasePersistence::PersistentDatabasePersistence(
    const std::string& vector_storage_path,
    std::shared_ptr<ShardingService> sharding_service,
    std::shared_ptr<QueryRouter> query_router,
    std::shared_ptr<ReplicationService> replication_service)
    : sharding_service_(sharding_service)
    , query_router_(query_router)
    , replication_service_(replication_service)
    , storage_path_(vector_storage_path) {
    
    logger_ = logging::LoggerManager::get_logger("PersistentDatabasePersistence");
    vector_store_ = std::make_unique<MemoryMappedVectorStore>(vector_storage_path);
    
    LOG_INFO(logger_, "Initialized PersistentDatabasePersistence with vector storage at: " << vector_storage_path);
    
    // Load existing databases from disk
    auto load_result = load_databases_from_disk();
    if (!load_result) {
        LOG_ERROR(logger_, "Failed to load databases from disk: " << ErrorHandler::format_error(load_result.error()));
    }
}

PersistentDatabasePersistence::~PersistentDatabasePersistence() = default;

Result<void> PersistentDatabasePersistence::save_database_metadata(const Database& db) {
    std::string metadata_file = storage_path_ + "/" + db.databaseId + "/metadata.json";
    
    try {
        std::ofstream file(metadata_file);
        if (!file.is_open()) {
            RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to create metadata file: " + metadata_file);
        }
        
        // Write simple JSON (we'll use manual formatting for simplicity)
        file << "{\n";
        file << "  \"database_id\": \"" << db.databaseId << "\",\n";
        file << "  \"name\": \"" << db.name << "\",\n";
        file << "  \"description\": \"" << db.description << "\",\n";
        file << "  \"vector_dimension\": " << db.vectorDimension << ",\n";
        file << "  \"index_type\": \"" << db.indexType << "\",\n";
        file << "  \"created_at\": \"" << db.created_at << "\",\n";
        file << "  \"updated_at\": \"" << db.updated_at << "\"\n";
        file << "}\n";
        
        file.close();
        return {};
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Exception saving metadata: " + std::string(e.what()));
    }
}

Result<Database> PersistentDatabasePersistence::load_database_metadata(const std::string& database_id) {
    std::string metadata_file = storage_path_ + "/" + database_id + "/metadata.json";
    
    try {
        std::ifstream file(metadata_file);
        if (!file.is_open()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Metadata file not found: " + metadata_file);
        }
        
        Database db;
        std::string line;
        
        // Simple JSON parsing (just look for key-value pairs)
        while (std::getline(file, line)) {
            // Remove whitespace and quotes
            size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            
            std::string key = line.substr(0, colon);
            std::string value = line.substr(colon + 1);
            
            // Trim whitespace and punctuation
            auto trim = [](std::string& s) {
                s.erase(0, s.find_first_not_of(" \t\n\r\f\v\""));
                s.erase(s.find_last_not_of(" \t\n\r\f\v\",") + 1);
            };
            trim(key);
            trim(value);
            
            if (key == "database_id") db.databaseId = value;
            else if (key == "name") db.name = value;
            else if (key == "description") db.description = value;
            else if (key == "vector_dimension") db.vectorDimension = std::stoi(value);
            else if (key == "index_type") db.indexType = value;
            else if (key == "created_at") db.created_at = value;
            else if (key == "updated_at") db.updated_at = value;
        }
        
        file.close();
        
        if (db.databaseId.empty() || db.name.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid metadata in file: " + metadata_file);
        }
        
        return db;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Exception loading metadata: " + std::string(e.what()));
    }
}

Result<void> PersistentDatabasePersistence::load_databases_from_disk() {
    LOG_INFO(logger_, "Loading databases from disk...");
    
    if (!std::filesystem::exists(storage_path_)) {
        LOG_WARN(logger_, "Storage path does not exist: " << storage_path_);
        return {};
    }
    
    int loaded_count = 0;
    int failed_count = 0;
    
    try {
        // Iterate through database directories
        for (const auto& entry : std::filesystem::directory_iterator(storage_path_)) {
            if (!entry.is_directory()) continue;
            
            std::string database_id = entry.path().filename().string();
            std::string vector_file = entry.path().string() + "/vectors.jvdb";
            std::string metadata_file = entry.path().string() + "/metadata.json";
            
            // Check if both files exist
            if (!std::filesystem::exists(vector_file)) {
                LOG_WARN(logger_, "Vector file not found for database " << database_id);
                failed_count++;
                continue;
            }
            
            if (!std::filesystem::exists(metadata_file)) {
                LOG_WARN(logger_, "Metadata file not found for database " << database_id);
                failed_count++;
                continue;
            }
            
            // Load metadata
            auto metadata_result = load_database_metadata(database_id);
            if (!metadata_result) {
                LOG_ERROR(logger_, "Failed to load metadata for " << database_id << ": " 
                         << ErrorHandler::format_error(metadata_result.error()));
                failed_count++;
                continue;
            }
            
            // Open vector file
            if (!vector_store_->open_vector_file(database_id)) {
                LOG_ERROR(logger_, "Failed to open vector file for database: " << database_id);
                failed_count++;
                continue;
            }
            
            // Store in memory
            databases_[database_id] = *metadata_result;
            indexes_by_db_[database_id] = std::unordered_map<std::string, Index>();
            
            loaded_count++;
            LOG_INFO(logger_, "Loaded database: " << metadata_result->name << " (ID: " << database_id << ")");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception while scanning database directories: " << e.what());
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to scan storage directory: " + std::string(e.what()));
    }
    
    LOG_INFO(logger_, "Database loading complete. Loaded: " << loaded_count << ", Failed: " << failed_count);
    
    return {};
}

Result<std::string> PersistentDatabasePersistence::create_database(const Database& db) {
    std::unique_lock<std::shared_mutex> lock(databases_mutex_);
    
    if (!db.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid database configuration");
    }
    
    std::string database_id = db.databaseId.empty() ? generate_id() : db.databaseId;
    Database new_db = db;
    new_db.databaseId = database_id;
    new_db.created_at = "2025-12-17T00:00:00Z";
    new_db.updated_at = "2025-12-17T00:00:00Z";
    
    databases_[database_id] = new_db;
    indexes_by_db_[database_id] = std::unordered_map<std::string, Index>();
    
    // Create vector file for this database
    if (!vector_store_->create_vector_file(database_id, new_db.vectorDimension, 1000)) {
        LOG_ERROR(logger_, "Failed to create vector file for database: " << database_id);
        databases_.erase(database_id);
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to create vector storage file");
    }
    
    // Save metadata to JSON file
    auto metadata_result = save_database_metadata(new_db);
    if (!metadata_result) {
        LOG_ERROR(logger_, "Failed to save database metadata: " << ErrorHandler::format_error(metadata_result.error()));
        // Continue - vector file is created, metadata can be synced later
    }
    
    LOG_INFO(logger_, "Created database with persistent storage: " << database_id);
    return database_id;
}

Result<Database> PersistentDatabasePersistence::get_database(const std::string& database_id) {
    std::shared_lock<std::shared_mutex> lock(databases_mutex_);
    
    auto it = databases_.find(database_id);
    if (it == databases_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    return it->second;
}

Result<std::vector<Database>> PersistentDatabasePersistence::list_databases() {
    std::shared_lock<std::shared_mutex> lock(databases_mutex_);
    
    std::vector<Database> result;
    for (const auto& [id, db] : databases_) {
        result.push_back(db);
    }
    
    return result;
}

Result<void> PersistentDatabasePersistence::update_database(const std::string& database_id, const Database& db) {
    std::unique_lock<std::shared_mutex> lock(databases_mutex_);
    
    if (!db.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid database configuration");
    }
    
    auto it = databases_.find(database_id);
    if (it == databases_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    Database updated_db = db;
    updated_db.databaseId = database_id;
    updated_db.updated_at = "2025-12-17T00:00:00Z";
    
    databases_[database_id] = updated_db;
    
    LOG_INFO(logger_, "Updated database: " << database_id);
    return {};
}

Result<void> PersistentDatabasePersistence::delete_database(const std::string& database_id) {
    std::unique_lock<std::shared_mutex> lock(databases_mutex_);
    
    auto it = databases_.find(database_id);
    if (it == databases_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    // Delete vector storage
    if (!vector_store_->delete_database_vectors(database_id)) {
        LOG_WARN(logger_, "Failed to delete vector storage for database: " << database_id);
    }
    
    databases_.erase(it);
    
    // Clean up indexes
    auto idx_it = indexes_by_db_.find(database_id);
    if (idx_it != indexes_by_db_.end()) {
        indexes_by_db_.erase(idx_it);
    }
    
    LOG_INFO(logger_, "Deleted database: " << database_id);
    return {};
}

Result<void> PersistentDatabasePersistence::store_vector(const std::string& database_id, const Vector& vector) {
    // Check database exists
    {
        std::shared_lock<std::shared_mutex> lock(databases_mutex_);
        if (databases_.find(database_id) == databases_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
        }
        
        // Validate dimensions
        const auto& db = databases_[database_id];
        if (!validate_vector_dimensions(vector, db.vectorDimension)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, 
                        "Vector dimension mismatch. Expected: " + std::to_string(db.vectorDimension) +
                        ", Got: " + std::to_string(vector.values.size()));
        }
    }
    
    // Store in memory-mapped file
    if (!vector_store_->store_vector(database_id, vector.id, vector.values)) {
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to store vector in persistent storage");
    }
    
    LOG_DEBUG(logger_, "Stored vector " << vector.id << " in database " << database_id);
    return {};
}

Result<Vector> PersistentDatabasePersistence::retrieve_vector(const std::string& database_id, const std::string& vector_id) {
    // Check database exists
    {
        std::shared_lock<std::shared_mutex> lock(databases_mutex_);
        if (databases_.find(database_id) == databases_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
        }
    }
    
    // Retrieve from memory-mapped file
    auto values = vector_store_->retrieve_vector(database_id, vector_id);
    if (!values.has_value()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Vector not found: " + vector_id);
    }
    
    // Construct Vector object
    Vector vec;
    vec.id = vector_id;
    vec.values = *values;
    vec.databaseId = database_id;
    
    return vec;
}

Result<std::vector<Vector>> PersistentDatabasePersistence::retrieve_vectors(
    const std::string& database_id,
    const std::vector<std::string>& vector_ids) {
    
    // Check database exists
    {
        std::shared_lock<std::shared_mutex> lock(databases_mutex_);
        if (databases_.find(database_id) == databases_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
        }
    }
    
    // Batch retrieve from memory-mapped file
    auto results = vector_store_->batch_retrieve(database_id, vector_ids);
    
    std::vector<Vector> vectors;
    for (size_t i = 0; i < results.size(); i++) {
        if (results[i].has_value()) {
            Vector vec;
            vec.id = vector_ids[i];
            vec.values = *results[i];
            vec.databaseId = database_id;
            vectors.push_back(vec);
        }
    }
    
    return vectors;
}

Result<void> PersistentDatabasePersistence::update_vector(const std::string& database_id, const Vector& vector) {
    // Update is same as store for memory-mapped storage
    return store_vector(database_id, vector);
}

Result<void> PersistentDatabasePersistence::delete_vector(const std::string& database_id, const std::string& vector_id) {
    // Check database exists
    {
        std::shared_lock<std::shared_mutex> lock(databases_mutex_);
        if (databases_.find(database_id) == databases_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
        }
    }
    
    if (!vector_store_->delete_vector(database_id, vector_id)) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Vector not found: " + vector_id);
    }
    
    LOG_DEBUG(logger_, "Deleted vector " << vector_id << " from database " << database_id);
    return {};
}

Result<void> PersistentDatabasePersistence::batch_store_vectors(
    const std::string& database_id,
    const std::vector<Vector>& vectors) {
    
    // Check database exists and get dimension
    int expected_dimension;
    {
        std::shared_lock<std::shared_mutex> lock(databases_mutex_);
        auto it = databases_.find(database_id);
        if (it == databases_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
        }
        expected_dimension = it->second.vectorDimension;
    }
    
    // Validate all vectors
    for (const auto& vec : vectors) {
        if (!validate_vector_dimensions(vec, expected_dimension)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT,
                        "Vector dimension mismatch for vector: " + vec.id);
        }
    }
    
    // Prepare batch
    std::vector<std::pair<std::string, std::vector<float>>> batch;
    for (const auto& vec : vectors) {
        batch.emplace_back(vec.id, vec.values);
    }
    
    // Batch store
    size_t stored = vector_store_->batch_store(database_id, batch);
    if (stored != vectors.size()) {
        LOG_WARN(logger_, "Only stored " << stored << " out of " << vectors.size() << " vectors");
    }
    
    LOG_INFO(logger_, "Batch stored " << stored << " vectors in database " << database_id);
    return {};
}

Result<void> PersistentDatabasePersistence::batch_delete_vectors(
    const std::string& database_id,
    const std::vector<std::string>& vector_ids) {
    
    // Check database exists
    {
        std::shared_lock<std::shared_mutex> lock(databases_mutex_);
        if (databases_.find(database_id) == databases_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
        }
    }
    
    size_t deleted = 0;
    for (const auto& vector_id : vector_ids) {
        if (vector_store_->delete_vector(database_id, vector_id)) {
            deleted++;
        }
    }
    
    LOG_INFO(logger_, "Batch deleted " << deleted << " vectors from database " << database_id);
    return {};
}

// Index operations (keep in-memory for now, can be extended later)
Result<void> PersistentDatabasePersistence::create_index(const std::string& database_id, const Index& index) {
    std::unique_lock<std::shared_mutex> lock(indexes_mutex_);
    
    auto& indexes = indexes_by_db_[database_id];
    std::string index_id = index.indexId.empty() ? generate_id() : index.indexId;
    
    Index new_index = index;
    new_index.indexId = index_id;
    indexes[index_id] = new_index;
    
    LOG_INFO(logger_, "Created index " << index_id << " for database " << database_id);
    return {};
}

Result<Index> PersistentDatabasePersistence::get_index(const std::string& database_id, const std::string& index_id) {
    std::shared_lock<std::shared_mutex> lock(indexes_mutex_);
    
    auto db_it = indexes_by_db_.find(database_id);
    if (db_it == indexes_by_db_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    auto idx_it = db_it->second.find(index_id);
    if (idx_it == db_it->second.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Index not found: " + index_id);
    }
    
    return idx_it->second;
}

Result<std::vector<Index>> PersistentDatabasePersistence::list_indexes(const std::string& database_id) {
    std::shared_lock<std::shared_mutex> lock(indexes_mutex_);
    
    auto it = indexes_by_db_.find(database_id);
    if (it == indexes_by_db_.end()) {
        return std::vector<Index>();
    }
    
    std::vector<Index> result;
    for (const auto& [id, idx] : it->second) {
        result.push_back(idx);
    }
    
    return result;
}

Result<void> PersistentDatabasePersistence::update_index(const std::string& database_id, const std::string& index_id, const Index& index) {
    std::unique_lock<std::shared_mutex> lock(indexes_mutex_);
    
    auto& indexes = indexes_by_db_[database_id];
    auto it = indexes.find(index_id);
    if (it == indexes.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Index not found: " + index_id);
    }
    
    Index updated_index = index;
    updated_index.indexId = index_id;
    indexes[index_id] = updated_index;
    
    LOG_INFO(logger_, "Updated index " << index_id << " in database " << database_id);
    return {};
}

Result<void> PersistentDatabasePersistence::delete_index(const std::string& database_id, const std::string& index_id) {
    std::unique_lock<std::shared_mutex> lock(indexes_mutex_);
    
    auto db_it = indexes_by_db_.find(database_id);
    if (db_it == indexes_by_db_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    auto it = db_it->second.find(index_id);
    if (it == db_it->second.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Index not found: " + index_id);
    }
    
    db_it->second.erase(it);
    
    LOG_INFO(logger_, "Deleted index " << index_id << " from database " << database_id);
    return {};
}

Result<bool> PersistentDatabasePersistence::database_exists(const std::string& database_id) const {
    std::shared_lock<std::shared_mutex> lock(databases_mutex_);
    return databases_.find(database_id) != databases_.end();
}

Result<bool> PersistentDatabasePersistence::vector_exists(const std::string& database_id, const std::string& vector_id) const {
    auto vec = const_cast<PersistentDatabasePersistence*>(this)->retrieve_vector(database_id, vector_id);
    return vec.has_value();
}

Result<bool> PersistentDatabasePersistence::index_exists(const std::string& database_id, const std::string& index_id) const {
    std::shared_lock<std::shared_mutex> lock(indexes_mutex_);
    
    auto db_it = indexes_by_db_.find(database_id);
    if (db_it == indexes_by_db_.end()) {
        return false;
    }
    
    return db_it->second.find(index_id) != db_it->second.end();
}

Result<size_t> PersistentDatabasePersistence::get_vector_count(const std::string& database_id) const {
    {
        std::shared_lock<std::shared_mutex> lock(databases_mutex_);
        if (databases_.find(database_id) == databases_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
        }
    }
    
    return vector_store_->get_vector_count(database_id);
}

Result<std::vector<std::string>> PersistentDatabasePersistence::get_all_vector_ids(const std::string& database_id) const {
    return vector_store_->list_vector_ids(database_id);
}

void PersistentDatabasePersistence::flush_all() {
    vector_store_->flush_all(false);
    LOG_INFO(logger_, "Flushed all vector stores");
}

void PersistentDatabasePersistence::flush_database(const std::string& database_id) {
    vector_store_->flush(database_id, false);
    LOG_DEBUG(logger_, "Flushed vector store for database: " << database_id);
}

std::string PersistentDatabasePersistence::generate_id() const {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    
    std::stringstream ss;
    ss << std::hex << std::setw(16) << std::setfill('0') << dis(gen);
    return ss.str();
}

bool PersistentDatabasePersistence::validate_vector_dimensions(const Vector& vector, int expected_dimension) const {
    return vector.values.size() == static_cast<size_t>(expected_dimension);
}

} // namespace jadevectordb