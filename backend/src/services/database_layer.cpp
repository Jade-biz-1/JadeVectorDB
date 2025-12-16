#include "database_layer.h"
#include "lib/logging.h"
#include "services/sharding_service.h"
#include "services/replication_service.h"
#include "services/query_router.h"
#include <random>
#include <algorithm>
#include <sstream>
#include <iomanip>

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

} // namespace jadevectordb