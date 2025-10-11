#include "vector_storage.h"
#include "lib/logging.h"
#include "lib/error_handling.h"

namespace jadevectordb {

VectorStorageService::VectorStorageService(std::unique_ptr<DatabaseLayer> db_layer)
    : db_layer_(std::move(db_layer)) {
    
    if (!db_layer_) {
        // If no database layer is provided, create a default one
        db_layer_ = std::make_unique<DatabaseLayer>();
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

Result<void> VectorStorageService::store_vector(const std::string& database_id, const Vector& vector) {
    // Validate the vector before storing
    auto validation_result = validate_vector(database_id, vector);
    if (!validation_result.has_value()) {
        return validation_result;
    }
    
    auto result = db_layer_->store_vector(database_id, vector);
    if (result.has_value()) {
        LOG_DEBUG(logger_, "Stored vector: " << vector.id << " in database: " << database_id);
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
    
    auto result = db_layer_->batch_store_vectors(database_id, vectors);
    if (result.has_value()) {
        LOG_DEBUG(logger_, "Batch stored " << vectors.size() << " vectors in database: " << database_id);
    }
    
    return result;
}

Result<Vector> VectorStorageService::retrieve_vector(const std::string& database_id, 
                                                    const std::string& vector_id) const {
    return db_layer_->retrieve_vector(database_id, vector_id);
}

Result<std::vector<Vector>> VectorStorageService::retrieve_vectors(const std::string& database_id, 
                                                                 const std::vector<std::string>& vector_ids) const {
    return db_layer_->retrieve_vectors(database_id, vector_ids);
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
    // This would require adding a method to the database layer
    // For now, we'll return 0 - a proper implementation would require 
    // a count method in the database layer interface
    return 0;
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
    // This would require implementing a method in the database layer to get all vector IDs
    // For now, we'll return an empty vector - a proper implementation would require 
    // a method to fetch all vector IDs in the database layer
    return std::vector<std::string>{};
}

} // namespace jadevectordb