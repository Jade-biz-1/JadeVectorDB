#include "database_service.h"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cctype>

namespace jadevectordb {

DatabaseService::DatabaseService(
    std::unique_ptr<DatabaseLayer> db_layer,
    std::shared_ptr<ClusterService> cluster_service,
    std::shared_ptr<ShardingService> sharding_service)
    : db_layer_(std::move(db_layer))
    , cluster_service_(cluster_service)
    , sharding_service_(sharding_service) {
    
    if (!db_layer_) {
        // If no database layer is provided, create a default one
        db_layer_ = std::make_unique<DatabaseLayer>();
    }
    
    logger_ = logging::LoggerManager::get_logger("DatabaseService");
}

Result<void> DatabaseService::initialize() {
    auto result = db_layer_->initialize();
    if (!result.has_value()) {
        LOG_ERROR(logger_, "Failed to initialize database layer: " << 
                 ErrorHandler::format_error(result.error()));
        return result;
    }
    
    LOG_INFO(logger_, "DatabaseService initialized successfully");
    return {};
}

Result<std::string> DatabaseService::create_database(const DatabaseCreationParams& params) {
    // Validate creation parameters
    auto validation_result = validate_creation_params(params);
    if (!validation_result.has_value()) {
        LOG_ERROR(logger_, "Database creation validation failed: " << 
                 ErrorHandler::format_error(validation_result.error()));
        return std::string{}; // Return empty string on validation failure
    }
    
    // Convert parameters to database object
    Database database = convert_params_to_database(params);
    
    // Generate database ID
    database.databaseId = generate_database_id();
    
    // Set timestamps
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&now_time_t), "%Y-%m-%dT%H:%M:%SZ");
    database.created_at = ss.str();
    database.updated_at = ss.str();
    
    // Store the database using the database layer
    auto result = db_layer_->create_database(database);
    if (!result.has_value()) {
        LOG_ERROR(logger_, "Failed to create database: " << 
                 ErrorHandler::format_error(result.error()));
        return std::string{};
    }
    
    std::string database_id = result.value();
    LOG_INFO(logger_, "Created database with ID: " << database_id << " (Name: " << params.name << ")");
    
    return database_id;
}

Result<Database> DatabaseService::get_database(const std::string& database_id) const {
    auto result = db_layer_->get_database(database_id);
    if (!result) {
        LOG_ERROR(logger_, "Failed to get database " << database_id << ": " << 
                 ErrorHandler::format_error(result.error()));
        return result;
    }
    
    LOG_DEBUG(logger_, "Retrieved database: " << database_id);
    return result;
}

Result<std::vector<Database>> DatabaseService::list_databases(const DatabaseListParams& params) const {
    // Get all databases first
    auto all_databases_result = db_layer_->list_databases();
    if (!all_databases_result.has_value()) {
        LOG_ERROR(logger_, "Failed to list databases: " << 
                 ErrorHandler::format_error(all_databases_result.error()));
        return std::vector<Database>{};
    }
    
    auto all_databases = all_databases_result.value();
    
    // Apply filtering
    std::vector<Database> filtered_databases;
    filtered_databases.reserve(all_databases.size());
    
    for (const auto& database : all_databases) {
        bool include = true;
        
        // Filter by name if specified
        if (!params.filterByName.empty()) {
            if (database.name.find(params.filterByName) == std::string::npos) {
                include = false;
            }
        }
        
        // Filter by owner if specified (would be in access control)
        if (include && !params.filterByOwner.empty()) {
            bool owner_found = false;
            for (const auto& role : database.accessControl.roles) {
                if (role == params.filterByOwner) {
                    owner_found = true;
                    break;
                }
            }
            if (!owner_found) {
                include = false;
            }
        }
        
        if (include) {
            filtered_databases.push_back(database);
        }
    }
    
    // Apply pagination
    size_t start_index = std::min(static_cast<size_t>(params.offset), filtered_databases.size());
    size_t end_index = std::min(start_index + static_cast<size_t>(params.limit), filtered_databases.size());
    
    if (start_index < filtered_databases.size()) {
        std::vector<Database> paginated_databases(
            filtered_databases.begin() + start_index,
            filtered_databases.begin() + end_index
        );
        return paginated_databases;
    }
    
    return std::vector<Database>();
}

Result<void> DatabaseService::update_database(const std::string& database_id, const DatabaseUpdateParams& params) {
    // Validate update parameters
    auto validation_result = validate_update_params(params);
    if (!validation_result.has_value()) {
        LOG_ERROR(logger_, "Database update validation failed: " << 
                 ErrorHandler::format_error(validation_result.error()));
        return {};
    }
    
    // Get the existing database
    auto get_result = db_layer_->get_database(database_id);
    if (!get_result.has_value()) {
        LOG_ERROR(logger_, "Failed to get database for update " << database_id << ": " << 
                 ErrorHandler::format_error(get_result.error()));
        return {};
    }
    
    Database& database = get_result.value();
    
    // Apply update parameters
    apply_update_params_to_database(database, params);
    
    // Update timestamp
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&now_time_t), "%Y-%m-%dT%H:%M:%SZ");
    database.updated_at = ss.str();
    
    // Update the database
    auto update_result = db_layer_->update_database(database_id, database);
    if (!update_result.has_value()) {
        LOG_ERROR(logger_, "Failed to update database " << database_id << ": " << 
                 ErrorHandler::format_error(update_result.error()));
        return {};
    }
    
    LOG_INFO(logger_, "Updated database: " << database_id);
    return {};
}

Result<void> DatabaseService::delete_database(const std::string& database_id) {
    auto result = db_layer_->delete_database(database_id);
    if (!result.has_value()) {
        LOG_ERROR(logger_, "Failed to delete database " << database_id << ": " << 
                 ErrorHandler::format_error(result.error()));
        return {};
    }
    
    LOG_INFO(logger_, "Deleted database: " << database_id);
    return {};
}

Result<bool> DatabaseService::database_exists(const std::string& database_id) const {
    auto result = db_layer_->database_exists(database_id);
    if (!result.has_value()) {
        LOG_ERROR(logger_, "Failed to check database existence " << database_id << ": " << 
                 ErrorHandler::format_error(result.error()));
        return false;
    }
    
    return result.value();
}

Result<size_t> DatabaseService::get_database_count() const {
    auto result = db_layer_->list_databases();
    if (!result.has_value()) {
        LOG_ERROR(logger_, "Failed to get database count: " << 
                 ErrorHandler::format_error(result.error()));
        return 0;
    }
    
    return result.value().size();
}

Result<void> DatabaseService::validate_creation_params(const DatabaseCreationParams& params) const {
    if (!params.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid database creation parameters");
    }
    
    // Additional validation
    if (params.name.length() > 255) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Database name too long (max 255 characters)");
    }
    
    if (params.description.length() > 1000) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Database description too long (max 1000 characters)");
    }
    
    // Validate index type
    std::vector<std::string> valid_index_types = {"HNSW", "IVF", "LSH", "FLAT"};
    if (std::find(valid_index_types.begin(), valid_index_types.end(), params.indexType) == valid_index_types.end()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid index type: " + params.indexType);
    }
    
    // Validate sharding configuration
    if (params.sharding.numShards <= 0) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Number of shards must be positive");
    }
    
    if (params.sharding.numShards > 1000) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Too many shards (max 1000)");
    }
    
    // Validate replication configuration
    if (params.replication.factor <= 0) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Replication factor must be positive");
    }
    
    if (params.replication.factor > 10) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Replication factor too high (max 10)");
    }
    
    return {};
}

Result<void> DatabaseService::validate_update_params(const DatabaseUpdateParams& params) const {
    // Validate individual parameters if they are set
    if (params.name.has_value() && params.name.value().length() > 255) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Database name too long (max 255 characters)");
    }
    
    if (params.description.has_value() && params.description.value().length() > 1000) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Database description too long (max 1000 characters)");
    }
    
    if (params.indexType.has_value()) {
        std::vector<std::string> valid_index_types = {"HNSW", "IVF", "LSH", "FLAT"};
        if (std::find(valid_index_types.begin(), valid_index_types.end(), params.indexType.value()) == valid_index_types.end()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid index type: " + params.indexType.value());
        }
    }
    
    if (params.sharding.has_value() && params.sharding.value().numShards <= 0) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Number of shards must be positive");
    }
    
    if (params.sharding.has_value() && params.sharding.value().numShards > 1000) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Too many shards (max 1000)");
    }
    
    if (params.replication.has_value() && params.replication.value().factor <= 0) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Replication factor must be positive");
    }
    
    if (params.replication.has_value() && params.replication.value().factor > 10) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Replication factor too high (max 10)");
    }
    
    return {};
}

Result<std::unordered_map<std::string, std::string>> DatabaseService::get_database_stats(const std::string& database_id) const {
    auto result = db_layer_->get_database(database_id);
    if (!result.has_value()) {
        LOG_ERROR(logger_, "Failed to get database for stats " << database_id << ": " << 
                 ErrorHandler::format_error(result.error()));
        RETURN_ERROR(ErrorCode::DATABASE_NOT_FOUND, "Database not found: " + database_id);
    }
    
    const auto& database = result.value();
    std::unordered_map<std::string, std::string> stats;
    
    stats["database_id"] = database.databaseId;
    stats["name"] = database.name;
    stats["vector_dimension"] = std::to_string(database.vectorDimension);
    stats["index_type"] = database.indexType;
    stats["shard_count"] = std::to_string(database.sharding.numShards);
    stats["replication_factor"] = std::to_string(database.replication.factor);
    stats["embedding_model_count"] = std::to_string(database.embeddingModels.size());
    stats["created_at"] = database.created_at;
    stats["updated_at"] = database.updated_at;
    
    return stats;
}

std::string DatabaseService::generate_database_id() const {
    // Generate a unique database ID
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto count = duration.count();
    
    std::stringstream ss;
    ss << "db_" << std::hex << count;
    return ss.str();
}

Database DatabaseService::convert_params_to_database(const DatabaseCreationParams& params) const {
    Database database;
    database.name = params.name;
    database.description = params.description;
    database.vectorDimension = params.vectorDimension;
    database.indexType = params.indexType;
    
    // Convert unordered_map to map for indexParameters
    for (const auto& [key, value] : params.indexParameters) {
        database.indexParameters[key] = value;
    }
    
    database.sharding = params.sharding;
    database.replication = params.replication;
    database.embeddingModels = params.embeddingModels;
    
    // Convert unordered_map to map for metadataSchema
    for (const auto& [key, value] : params.metadataSchema) {
        database.metadataSchema[key] = value;
    }
    
    if (params.retentionPolicy) {
        database.retentionPolicy = std::make_unique<Database::RetentionPolicy>(*params.retentionPolicy);
    }
    
    database.accessControl = params.accessControl;
    
    return database;
}

void DatabaseService::apply_update_params_to_database(Database& database, const DatabaseUpdateParams& params) const {
    if (params.name.has_value()) {
        database.name = params.name.value();
    }
    
    if (params.description.has_value()) {
        database.description = params.description.value();
    }
    
    if (params.vectorDimension.has_value()) {
        database.vectorDimension = params.vectorDimension.value();
    }
    
    if (params.indexType.has_value()) {
        database.indexType = params.indexType.value();
    }
    
    if (params.indexParameters.has_value()) {
        // Convert unordered_map to map
        database.indexParameters.clear();
        for (const auto& [key, value] : params.indexParameters.value()) {
            database.indexParameters[key] = value;
        }
    }
    
    if (params.sharding.has_value()) {
        database.sharding = params.sharding.value();
    }
    
    if (params.replication.has_value()) {
        database.replication = params.replication.value();
    }
    
    if (params.embeddingModels.has_value()) {
        database.embeddingModels = params.embeddingModels.value();
    }
    
    if (params.metadataSchema.has_value()) {
        // Convert unordered_map to map
        database.metadataSchema.clear();
        for (const auto& [key, value] : params.metadataSchema.value()) {
            database.metadataSchema[key] = value;
        }
    }
    
    if (params.retentionPolicy.has_value()) {
        if (params.retentionPolicy.value()) {
            database.retentionPolicy = std::make_unique<Database::RetentionPolicy>(*params.retentionPolicy.value());
        } else {
            database.retentionPolicy.reset();
        }
    }
    
    if (params.accessControl.has_value()) {
        database.accessControl = params.accessControl.value();
    }
}

} // namespace jadevectordb