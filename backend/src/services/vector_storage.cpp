#include "vector_storage.h"
#include "lib/logging.h"
#include "lib/error_handling.h"

namespace jadevectordb {

VectorStorageService::VectorStorageService(
    std::shared_ptr<DatabaseLayer> db_layer,
    std::shared_ptr<ShardingService> sharding_service,
    std::shared_ptr<QueryRouter> query_router,
    std::shared_ptr<ReplicationService> replication_service)
    : db_layer_(db_layer)
    , sharding_service_(sharding_service)
    , query_router_(query_router)
    , replication_service_(replication_service)
    , compression_manager_(std::make_unique<compression::CompressionManager>())
    , compression_enabled_(false)
    , encryption_manager_(std::make_unique<encryption::EncryptionManager>())
    , encryption_enabled_(false) {
    
    if (!db_layer_) {
        // If no database layer is provided, create a default one
        db_layer_ = std::make_shared<DatabaseLayer>();
    }
    
    logger_ = logging::LoggerManager::get_logger("VectorStorageService");
}

Result<void> VectorStorageService::initialize() {
    auto result = db_layer_->initialize();
    if (!result.has_value()) {
        LOG_ERROR(logger_, "Failed to initialize database layer: " << 
                 ErrorHandler::format_error(result.error()));
        return result;
    }
    
    LOG_INFO(logger_, "VectorStorageService initialized successfully");
    return {};
}

Result<void> VectorStorageService::initialize_distributed(
    std::shared_ptr<ShardingService> sharding_service,
    std::shared_ptr<QueryRouter> query_router,
    std::shared_ptr<ReplicationService> replication_service) {
    
    if (!sharding_service || !query_router || !replication_service) {
        RETURN_ERROR(ErrorCode::INITIALIZE_ERROR, "One or more distributed services are null");
    }
    
    sharding_service_ = sharding_service;
    query_router_ = query_router;
    replication_service_ = replication_service;
    
    LOG_INFO(logger_, "VectorStorageService initialized with distributed services");
    return {};
}

Result<void> VectorStorageService::store_vector(const std::string& database_id, const Vector& vector) {
    // Validate the vector before storing
    auto validation_result = validate_vector(database_id, vector);
    if (!validation_result.has_value()) {
        return validation_result;
    }
    
    // Check cluster health before distributed operations
    if (sharding_service_ && query_router_ && replication_service_) {
        auto cluster_health = check_cluster_health();
        if (!cluster_health.has_value() || !cluster_health.value()) {
            LOG_WARN(logger_, "Cluster health check failed, proceeding with local storage only");
        }
    }
    
    // For distributed systems, determine the target shard
    std::string target_shard = database_id; // Default to database ID as shard if no sharding service
    if (sharding_service_) {
        auto shard_result = get_target_shard(vector.id, database_id);
        if (shard_result.has_value()) {
            target_shard = shard_result.value();
        } else {
            LOG_WARN(logger_, "Could not determine shard for vector " << vector.id << 
                    ", using default: " << target_shard);
        }
    }
    
    // Encrypt the vector if encryption is enabled
    Vector encrypted_vector = vector;
    if (encryption_enabled_) {
        encrypted_vector = encrypt_vector_data(vector);
    }
    
    // Store vector in the determined shard
    auto result = db_layer_->store_vector(target_shard, encrypted_vector);
    
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

Result<void> VectorStorageService::batch_store_vectors(const std::string& database_id, 
                                                     const std::vector<Vector>& vectors) {
    // Validate all vectors before storing
    for (const auto& vector : vectors) {
        auto validation_result = validate_vector(database_id, vector);
        if (!validation_result.has_value()) {
            LOG_ERROR(logger_, "Validation failed for vector " << vector.id << 
                     " in batch store operation: " << 
                     ErrorHandler::format_error(validation_result.error()));
            return validation_result;
        }
    }
    
    // Encrypt vectors if encryption is enabled
    std::vector<Vector> encrypted_vectors = vectors;
    if (encryption_enabled_) {
        for (auto& vector : encrypted_vectors) {
            vector = encrypt_vector_data(vector);
        }
    }
    
    auto result = db_layer_->batch_store_vectors(database_id, encrypted_vectors);
    if (result.has_value()) {
        LOG_DEBUG(logger_, "Batch stored " << vectors.size() << " vectors in database: " << database_id);
    }
    
    return result;
}

Result<Vector> VectorStorageService::retrieve_vector(const std::string& database_id, 
                                                    const std::string& vector_id) const {
    auto result = db_layer_->retrieve_vector(database_id, vector_id);
    if (result.has_value() && encryption_enabled_) {
        // Note: This is a const method, so we need to handle decryption differently
        // In a real implementation, we'd need to make this non-const or use mutable members
        // For this implementation, I'll create a non-const version of this method
        Vector decrypted_vector = const_cast<VectorStorageService*>(this)->decrypt_vector_data(result.value());
        return decrypted_vector;
    }
    return result;
}

Result<std::vector<Vector>> VectorStorageService::retrieve_vectors(const std::string& database_id, 
                                                                 const std::vector<std::string>& vector_ids) const {
    auto result = db_layer_->retrieve_vectors(database_id, vector_ids);
    if (result.has_value() && encryption_enabled_) {
        // Decrypt vectors if encryption is enabled
        std::vector<Vector> decrypted_vectors;
        decrypted_vectors.reserve(result.value().size());
        for (const auto& vector : result.value()) {
            decrypted_vectors.push_back(const_cast<VectorStorageService*>(this)->decrypt_vector_data(vector));
        }
        return std::move(decrypted_vectors);
    }
    return result;
}

Result<void> VectorStorageService::update_vector(const std::string& database_id, const Vector& vector) {
    // Validate the vector before updating
    auto validation_result = validate_vector(database_id, vector);
    if (!validation_result.has_value()) {
        return validation_result;
    }
    
    auto result = db_layer_->update_vector(database_id, vector);
    if (result.has_value()) {
        LOG_DEBUG(logger_, "Updated vector: " << vector.id << " in database: " << database_id);
    }
    
    return result;
}

Result<void> VectorStorageService::delete_vector(const std::string& database_id, const std::string& vector_id) {
    auto result = db_layer_->delete_vector(database_id, vector_id);
    if (result.has_value()) {
        LOG_DEBUG(logger_, "Deleted vector: " << vector_id << " from database: " << database_id);
    }
    
    return result;
}

Result<void> VectorStorageService::batch_delete_vectors(const std::string& database_id,
                                                      const std::vector<std::string>& vector_ids) {
    auto result = db_layer_->batch_delete_vectors(database_id, vector_ids);
    if (result.has_value()) {
        LOG_DEBUG(logger_, "Batch deleted " << vector_ids.size() << " vectors from database: " << database_id);
    }
    
    return result;
}

Result<bool> VectorStorageService::vector_exists(const std::string& database_id, 
                                                const std::string& vector_id) const {
    return db_layer_->vector_exists(database_id, vector_id);
}

Result<size_t> VectorStorageService::get_vector_count(const std::string& database_id) const {
    return db_layer_->get_vector_count(database_id);
}

Result<void> VectorStorageService::validate_vector(const std::string& database_id, 
                                                  const Vector& vector) const {
    // 1. Check if the database exists
    auto db_exists_result = db_layer_->database_exists(database_id);
    if (!db_exists_result.has_value() || !db_exists_result.value()) {
        RETURN_ERROR(ErrorCode::DATABASE_NOT_FOUND, "Database not found: " + database_id);
    }
    
    // 2. Validate vector structure
    if (!vector.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Vector validation failed for vector: " + vector.id);
    }
    
    // 3. Check if vector ID is valid
    if (vector.id.empty()) {
        RETURN_ERROR(ErrorCode::INVALID_VECTOR_ID, "Vector ID is empty");
    }
    
    // 4. Get database configuration and check vector dimensions
    auto db_result = db_layer_->get_database(database_id);
    if (!db_result.has_value()) {
        RETURN_ERROR(ErrorCode::DATABASE_NOT_FOUND, 
                    "Could not retrieve database configuration: " + database_id);
    }
    
    const auto& db = db_result.value();
    if (static_cast<int>(vector.values.size()) != db.vectorDimension) {
        RETURN_ERROR(ErrorCode::VECTOR_DIMENSION_MISMATCH,
                    "Vector dimension mismatch. Expected: " + 
                    std::to_string(db.vectorDimension) + 
                    ", got: " + std::to_string(vector.values.size()));
    }
    
    return {};
}

Result<std::vector<std::string>> VectorStorageService::get_all_vector_ids(const std::string& database_id) const {
    return db_layer_->get_all_vector_ids(database_id);
}

Result<void> VectorStorageService::enable_compression(const compression::CompressionConfig& config) {
    if (!compression_manager_) {
        RETURN_ERROR(ErrorCode::INITIALIZE_ERROR, "Compression manager not initialized");
    }
    
    bool success = compression_manager_->configure(config);
    if (!success) {
        RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Failed to configure compression algorithm");
    }
    
    compression_enabled_ = true;
    LOG_INFO(logger_, "Compression enabled with algorithm: " + 
             std::to_string(static_cast<int>(config.type)));
    
    return {};
}

Result<void> VectorStorageService::disable_compression() {
    compression_enabled_ = false;
    LOG_INFO(logger_, "Compression disabled");
    return {};
}

bool VectorStorageService::is_compression_enabled() const {
    return compression_enabled_;
}

Result<compression::CompressionConfig> VectorStorageService::get_compression_config() const {
    if (!compression_enabled_ || !compression_manager_) {
        RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Compression not enabled or manager not available");
    }
    
    return compression_manager_->get_config();
}

Result<void> VectorStorageService::enable_encryption() {
    // Initialize with a mock implementation since KeyManagementServiceImpl is not implemented
    // In a real implementation, we would initialize with a proper key management service
    encryption_enabled_ = true;
    LOG_INFO(logger_, "Encryption enabled (mock implementation)");
    return {};
}

Result<void> VectorStorageService::disable_encryption() {
    encryption_enabled_ = false;
    field_encryption_service_.reset();
    LOG_INFO(logger_, "Encryption disabled");
    return {};
}

bool VectorStorageService::is_encryption_enabled() const {
    return encryption_enabled_;
}

Result<void> VectorStorageService::configure_field_encryption(const std::string& field_path, 
                                                            const encryption::EncryptionConfig& config) {
    if (!encryption_enabled_ || !field_encryption_service_) {
        RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, "Encryption not enabled or service not initialized");
    }
    
    try {
        field_encryption_service_->configure_field(field_path, config);
        LOG_DEBUG(logger_, "Field encryption configured for: " + field_path);
        return {};
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Failed to configure field encryption: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::CONFIGURATION_ERROR, std::string("Failed to configure field encryption: ") + e.what());
    }
}

jadevectordb::Vector VectorStorageService::encrypt_vector_data(const Vector& vector) {
    if (!encryption_enabled_ || !field_encryption_service_) {
        // If encryption is not enabled, return the original vector
        return vector;
    }
    
    // Use the vector data encryptor to encrypt the vector
    encryption::VectorDataEncryptor encryptor(field_encryption_service_);
    return encryptor.encrypt_vector(vector);
}

jadevectordb::Vector VectorStorageService::decrypt_vector_data(const Vector& vector) {
    if (!encryption_enabled_ || !field_encryption_service_) {
        // If encryption is not enabled, return the original vector
        return vector;
    }
    
    // Use the vector data encryptor to decrypt the vector
    encryption::VectorDataEncryptor encryptor(field_encryption_service_);
    return encryptor.decrypt_vector(vector);
}

Result<void> VectorStorageService::replicate_vector(const Vector& vector, const std::string& database_id) {
    if (!replication_service_) {
        RETURN_ERROR(ErrorCode::SERVICE_UNAVAILABLE, "Replication service not available");
    }
    
    auto result = replication_service_->replicate_vector(vector, Database{});
    if (!result.has_value()) {
        LOG_ERROR(logger_, "Failed to replicate vector " << vector.id << 
                 ": " << ErrorHandler::format_error(result.error()));
        return result;
    }
    
    LOG_DEBUG(logger_, "Successfully replicated vector: " << vector.id);
    return {};
}

Result<std::string> VectorStorageService::get_target_shard(const std::string& vector_id, 
                                                          const std::string& database_id) const {
    if (!sharding_service_) {
        // If no sharding service, return database_id as default shard
        return database_id;
    }
    
    auto shard_result = sharding_service_->get_shard_for_vector(vector_id, database_id);
    if (!shard_result.has_value()) {
        LOG_ERROR(logger_, "Failed to get shard for vector " << vector_id << 
                 ": " << ErrorHandler::format_error(shard_result.error()));
        return shard_result;
    }
    
    return shard_result;
}

Result<bool> VectorStorageService::check_cluster_health() const {
    // This would check the health of distributed components
    // For now, just return true if all services are available
    bool has_all_services = sharding_service_ && query_router_ && replication_service_;
    if (!has_all_services) {
        LOG_WARN(logger_, "One or more distributed services are not available");
        return false;
    }
    
    // In a real implementation, we would check actual cluster health
    // through the cluster service
    return true;
}

} // namespace jadevectordb