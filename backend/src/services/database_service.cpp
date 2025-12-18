#include "database_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>
#include <chrono>

namespace jadevectordb {

DatabaseService::DatabaseService(
    std::shared_ptr<DatabaseLayer> db_layer,
    std::shared_ptr<ClusterService> cluster_service,
    std::shared_ptr<ShardingService> sharding_service
) : db_layer_(db_layer),
    cluster_service_(cluster_service),
    sharding_service_(sharding_service) {
    logger_ = logging::LoggerManager::get_logger("DatabaseService");
}

Result<void> DatabaseService::initialize() {
    try {
        LOG_INFO(logger_, "Initializing DatabaseService");
        
        // Initialize database layer if not already initialized
        if (!db_layer_) {
            db_layer_ = std::make_shared<DatabaseLayer>();
            auto result = db_layer_->initialize();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to initialize database layer: " +
                         ErrorHandler::format_error(result.error()));
                return result;
            }
        }
        
        LOG_INFO(logger_, "DatabaseService initialized successfully");
        return {};
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in DatabaseService::initialize: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to initialize DatabaseService: " + std::string(e.what()));
    }
}

Result<std::string> DatabaseService::create_database(const DatabaseCreationParams& params) {
    try {
        LOG_INFO(logger_, "Creating database with name: " + params.name);
        
        // Validate database creation parameters
        auto validation_result = validate_creation_params(params);
        if (!validation_result.has_value()) {
            LOG_ERROR(logger_, "Database creation validation failed: " + 
                     ErrorHandler::format_error(validation_result.error()));
            return tl::make_unexpected(validation_result.error());
        }
        
        // Create a new database object
        Database database;
        database.databaseId = generate_database_id();
        database.name = params.name;
        database.description = params.description;
        database.vectorDimension = params.vectorDimension;
        database.indexType = params.indexType;
        database.indexParameters = params.indexParameters;
        database.sharding = params.sharding;
        database.replication = params.replication;
        database.embeddingModels = params.embeddingModels;
        database.metadataSchema = params.metadataSchema;
        database.accessControl = params.accessControl;
        database.created_at = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        database.updated_at = database.created_at;
        
        // Handle unique_ptr assignment
        if (params.retentionPolicy) {
            database.retentionPolicy = std::make_unique<Database::RetentionPolicy>(*params.retentionPolicy);
        }
        
        // Store the database using the database layer
        auto result = db_layer_->create_database(database);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to create database in database layer: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_INFO(logger_, "Database created successfully with ID: " + database.databaseId);
        return database.databaseId;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in create_database: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to create database: " + std::string(e.what()));
    }
}

Result<Database> DatabaseService::get_database(const std::string& database_id) const {
    try {
        LOG_DEBUG(logger_, "Getting database with ID: " + database_id);
        
        // Retrieve database from database layer
        auto result = db_layer_->get_database(database_id);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to get database " + database_id + 
                     " from database layer: " + ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_DEBUG(logger_, "Retrieved database: " + database_id);
        return result.value();
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_database: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to get database: " + std::string(e.what()));
    }
}

Result<std::vector<Database>> DatabaseService::list_databases(const DatabaseListParams& params) const {
    try {
        LOG_DEBUG(logger_, "Listing databases");
        
        // Get all databases from the database layer
        auto result = db_layer_->list_databases();
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to list databases from database layer: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        auto databases = result.value();
        
        // Apply filtering and sorting based on params
        if (!params.filterByName.empty()) {
            databases.erase(
                std::remove_if(databases.begin(), databases.end(), 
                              [&params](const Database& db) { 
                                  return db.name.find(params.filterByName) == std::string::npos; 
                              }),
                databases.end()
            );
        }
        
        if (!params.filterByOwner.empty()) {
            databases.erase(
                std::remove_if(databases.begin(), databases.end(), 
                              [&params](const Database& db) { 
                                  return db.owner != params.filterByOwner; 
                              }),
                databases.end()
            );
        }
        
        // Sort by name if requested
        if (params.sortByName) {
            std::sort(databases.begin(), databases.end(), 
                     [](const Database& a, const Database& b) { 
                         return a.name < b.name; 
                     });
        }
        
        // Apply pagination
        if (params.limit > 0) {
            size_t start = std::min(static_cast<size_t>(params.offset), databases.size());
            size_t end = std::min(start + static_cast<size_t>(params.limit), databases.size());
            if (start < databases.size()) {
                databases = std::vector<Database>(databases.begin() + start, databases.begin() + end);
            } else {
                databases.clear();
            }
        }
        
        LOG_DEBUG(logger_, "Listed " + std::to_string(databases.size()) + " databases");
        return databases;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in list_databases: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to list databases: " + std::string(e.what()));
    }
}

Result<void> DatabaseService::update_database(const std::string& database_id, 
                                           const DatabaseUpdateParams& params) {
    try {
        LOG_INFO(logger_, "Updating database with ID: " + database_id);
        
        // Validate database update parameters
        auto validation_result = validate_update_params(params);
        if (!validation_result.has_value()) {
            LOG_ERROR(logger_, "Database update validation failed: " + 
                     ErrorHandler::format_error(validation_result.error()));
            return validation_result;
        }
        
        // Get current database
        auto current_db_result = get_database(database_id);
        if (!current_db_result.has_value()) {
            LOG_ERROR(logger_, "Failed to get current database for update: " + 
                     ErrorHandler::format_error(current_db_result.error()));
            return tl::make_unexpected(current_db_result.error());
        }
        
        Database updated_db = current_db_result.value();
        updated_db.updated_at = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Apply updates
        if (params.name.has_value()) {
            updated_db.name = params.name.value();
        }
        if (params.description.has_value()) {
            updated_db.description = params.description.value();
        }
        if (params.vectorDimension.has_value() && params.vectorDimension.value() > 0) {
            updated_db.vectorDimension = params.vectorDimension.value();
        }
        if (params.indexType.has_value()) {
            updated_db.indexType = params.indexType.value();
        }
        
        // Update database in database layer
        auto result = db_layer_->update_database(database_id, updated_db);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to update database in database layer: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_INFO(logger_, "Database updated successfully: " + database_id);
        return {};
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_database: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to update database: " + std::string(e.what()));
    }
}

Result<void> DatabaseService::delete_database(const std::string& database_id) {
    try {
        LOG_INFO(logger_, "Deleting database with ID: " + database_id);
        
        // Delete database from database layer
        auto result = db_layer_->delete_database(database_id);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to delete database from database layer: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_INFO(logger_, "Database deleted successfully: " + database_id);
        return {};
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in delete_database: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to delete database: " + std::string(e.what()));
    }
}

Result<bool> DatabaseService::database_exists(const std::string& database_id) const {
    try {
        LOG_DEBUG(logger_, "Checking if database exists: " + database_id);
        
        // Check if database exists in database layer
        auto result = db_layer_->database_exists(database_id);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to check database existence: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_DEBUG(logger_, "Database " + database_id + " exists: " + 
                 (result.value() ? "true" : "false"));
        return result.value();
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in database_exists: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to check database existence: " + std::string(e.what()));
    }
}

Result<void> DatabaseService::validate_creation_params(const DatabaseCreationParams& params) const {
    try {
        // Validate database name
        if (params.name.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Database name is required");
        }
        
        // Validate vector dimension
        if (params.vectorDimension <= 0) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Vector dimension must be positive");
        }
        
        if (params.vectorDimension > 10000) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Vector dimension is too large (max 10000)");
        }
        
        // Validate index type (case-insensitive)
        if (!params.indexType.empty()) {
            std::string index_type_lower = params.indexType;
            std::transform(index_type_lower.begin(), index_type_lower.end(), 
                          index_type_lower.begin(), ::tolower);
            
            std::vector<std::string> valid_index_types = {"hnsw", "ivf", "flat", "lsh"};
            if (std::find(valid_index_types.begin(), valid_index_types.end(), index_type_lower) == 
                valid_index_types.end()) {
                RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid index type: " + params.indexType + 
                           " (valid types: HNSW, IVF, FLAT, LSH)");
            }
        }
        
        LOG_DEBUG(logger_, "Database creation parameters validated successfully");
        return {};
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in validate_creation_params: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to validate database creation parameters: " + std::string(e.what()));
    }
}

Result<void> DatabaseService::validate_update_params(const DatabaseUpdateParams& params) const {
    try {
        // Validate vector dimension if provided
        if (params.vectorDimension.has_value() && params.vectorDimension.value() > 0 && params.vectorDimension.value() > 10000) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Vector dimension is too large (max 10000)");
        }
        
        // Validate index type if provided
        if (params.indexType.has_value() && !params.indexType.value().empty()) {
            std::vector<std::string> valid_index_types = {"hnsw", "ivf", "flat", "lsh"};
            if (std::find(valid_index_types.begin(), valid_index_types.end(), params.indexType.value()) == 
                valid_index_types.end()) {
                RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid index type: " + params.indexType.value());
            }
        }
        
        LOG_DEBUG(logger_, "Database update parameters validated successfully");
        return {};
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in validate_update_params: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to validate database update parameters: " + std::string(e.what()));
    }
}

std::string DatabaseService::generate_database_id() const {
    // Generate a unique database ID based on timestamp
    auto now = std::chrono::high_resolution_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    return "db_" + std::to_string(nanoseconds);
}

DatabaseService::DatabaseRole DatabaseService::get_role_for_database(const std::string& database_id) const {
    // In a distributed system, this would determine the role for a specific database
    // For now, we'll return MASTER as default
    return DatabaseRole::MASTER;
}

bool DatabaseService::is_master_for_database(const std::string& database_id) const {
    return get_role_for_database(database_id) == DatabaseRole::MASTER;
}

Result<std::vector<std::string>> DatabaseService::get_database_names() const {
    try {
        LOG_DEBUG(logger_, "Getting database names");
        
        // Get all databases and extract names
        auto databases_result = list_databases(DatabaseListParams{});
        if (!databases_result.has_value()) {
            LOG_ERROR(logger_, "Failed to list databases: " + 
                     ErrorHandler::format_error(databases_result.error()));
            return tl::make_unexpected(databases_result.error());
        }
        
        auto databases = databases_result.value();
        std::vector<std::string> names;
        names.reserve(databases.size());
        
        for (const auto& db : databases) {
            names.push_back(db.name);
        }
        
        LOG_DEBUG(logger_, "Retrieved " + std::to_string(names.size()) + " database names");
        return names;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_database_names: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to get database names: " + std::string(e.what()));
    }
}
}

