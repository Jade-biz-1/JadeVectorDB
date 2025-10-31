#include "database_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>
#include <chrono>

namespace jadevectordb {

DatabaseService::DatabaseService() {
    logger_ = logging::LoggerManager::get_logger("DatabaseService");
}

bool DatabaseService::initialize() {
    try {
        LOG_INFO(logger_, "Initializing DatabaseService");
        
        // Initialize database layer if not already initialized
        if (!db_layer_) {
            db_layer_ = std::make_unique<DatabaseLayer>();
            auto result = db_layer_->initialize();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to initialize database layer: " + 
                         ErrorHandler::format_error(result.error()));
                return false;
            }
        }
        
        LOG_INFO(logger_, "DatabaseService initialized successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in DatabaseService::initialize: " + std::string(e.what()));
        return false;
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
            return validation_result;
        }
        
        // Create a new database object
        Database database;
        database.databaseId = generate_database_id(params.name);
        database.name = params.name;
        database.description = params.description;
        database.vectorDimension = params.vectorDimension;
        database.indexType = params.indexType;
        database.created_at = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        database.updated_at = database.created_at;
        
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
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to create database: " + std::string(e.what()));
    }
}

Result<std::vector<Database>> DatabaseService::list_databases(const DatabaseListParams& params) {
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
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to list databases: " + std::string(e.what()));
    }
}

Result<Database> DatabaseService::get_database(const std::string& database_id) {
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
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get database: " + std::string(e.what()));
    }
}

Result<bool> DatabaseService::update_database(const std::string& database_id, 
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
            return current_db_result;
        }
        
        Database updated_db = current_db_result.value();
        updated_db.updated_at = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Apply updates
        if (!params.name.empty()) {
            updated_db.name = params.name;
        }
        if (!params.description.empty()) {
            updated_db.description = params.description;
        }
        if (params.vectorDimension > 0) {
            updated_db.vectorDimension = params.vectorDimension;
        }
        if (!params.indexType.empty()) {
            updated_db.indexType = params.indexType;
        }
        
        // Update database in database layer
        auto result = db_layer_->update_database(database_id, updated_db);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to update database in database layer: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_INFO(logger_, "Database updated successfully: " + database_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_database: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update database: " + std::string(e.what()));
    }
}

Result<bool> DatabaseService::delete_database(const std::string& database_id) {
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
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in delete_database: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to delete database: " + std::string(e.what()));
    }
}

Result<bool> DatabaseService::database_exists(const std::string& database_id) {
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
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check database existence: " + std::string(e.what()));
    }
}

Result<bool> DatabaseService::validate_creation_params(const DatabaseCreationParams& params) {
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
        
        // Validate index type
        if (!params.indexType.empty()) {
            std::vector<std::string> valid_index_types = {"hnsw", "ivf", "flat", "lsh"};
            if (std::find(valid_index_types.begin(), valid_index_types.end(), params.indexType) == 
                valid_index_types.end()) {
                RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid index type: " + params.indexType);
            }
        }
        
        LOG_DEBUG(logger_, "Database creation parameters validated successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in validate_creation_params: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to validate database creation parameters: " + std::string(e.what()));
    }
}

Result<bool> DatabaseService::validate_update_params(const DatabaseUpdateParams& params) {
    try {
        // Validate vector dimension if provided
        if (params.vectorDimension > 0 && params.vectorDimension > 10000) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Vector dimension is too large (max 10000)");
        }
        
        // Validate index type if provided
        if (!params.indexType.empty()) {
            std::vector<std::string> valid_index_types = {"hnsw", "ivf", "flat", "lsh"};
            if (std::find(valid_index_types.begin(), valid_index_types.end(), params.indexType) == 
                valid_index_types.end()) {
                RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid index type: " + params.indexType);
            }
        }
        
        LOG_DEBUG(logger_, "Database update parameters validated successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in validate_update_params: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to validate database update parameters: " + std::string(e.what()));
    }
}

std::string DatabaseService::generate_database_id(const std::string& database_name) {
    // Generate a unique database ID based on name and timestamp
    auto now = std::chrono::high_resolution_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    return database_name + "_" + std::to_string(nanoseconds);
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
            return databases_result;
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
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get database names: " + std::string(e.what()));
    }
}

Result<size_t> DatabaseService::get_database_count() const {
    try {
        LOG_DEBUG(logger_, "Getting database count");
        
        // Get all databases and count them
        auto databases_result = list_databases(DatabaseListParams{});
        if (!databases_result.has_value()) {
            LOG_ERROR(logger_, "Failed to list databases: " + 
                     ErrorHandler::format_error(databases_result.error()));
            return databases_result;
        }
        
        auto count = databases_result.value().size();
        LOG_DEBUG(logger_, "Database count: " + std::to_string(count));
        return count;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_database_count: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get database count: " + std::string(e.what()));
    }
}

Result<bool> DatabaseService::check_database_health(const std::string& database_id) const {
    try {
        LOG_DEBUG(logger_, "Checking health of database: " + database_id);
        
        // Check if database exists
        auto exists_result = database_exists(database_id);
        if (!exists_result.has_value()) {
            LOG_ERROR(logger_, "Failed to check database existence: " + 
                     ErrorHandler::format_error(exists_result.error()));
            return exists_result;
        }
        
        if (!exists_result.value()) {
            RETURN_ERROR(ErrorCode::DATABASE_NOT_FOUND, "Database not found: " + database_id);
        }
        
        // In a real implementation, we would check various health metrics
        // For now, we'll assume healthy if database exists
        LOG_DEBUG(logger_, "Database " + database_id + " is healthy");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in check_database_health: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check database health: " + std::string(e.what()));
    }
}

Result<std::unordered_map<std::string, std::string>> DatabaseService::get_database_stats(const std::string& database_id) const {
    try {
        LOG_DEBUG(logger_, "Getting stats for database: " + database_id);
        
        // Check if database exists
        auto exists_result = database_exists(database_id);
        if (!exists_result.has_value()) {
            LOG_ERROR(logger_, "Failed to check database existence: " + 
                     ErrorHandler::format_error(exists_result.error()));
            return exists_result;
        }
        
        if (!exists_result.value()) {
            RETURN_ERROR(ErrorCode::DATABASE_NOT_FOUND, "Database not found: " + database_id);
        }
        
        // Get database info
        auto db_result = get_database(database_id);
        if (!db_result.has_value()) {
            LOG_ERROR(logger_, "Failed to get database info: " + 
                     ErrorHandler::format_error(db_result.error()));
            return db_result;
        }
        
        auto db = db_result.value();
        
        std::unordered_map<std::string, std::string> stats;
        stats["databaseId"] = database_id;
        stats["name"] = db.name;
        stats["description"] = db.description;
        stats["vectorDimension"] = std::to_string(db.vectorDimension);
        stats["indexType"] = db.indexType;
        stats["created_at"] = std::to_string(db.created_at);
        stats["updated_at"] = std::to_string(db.updated_at);
        
        LOG_DEBUG(logger_, "Retrieved stats for database: " + database_id);
        return stats;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_database_stats: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get database stats: " + std::string(e.what()));
    }
}

} // namespace jadevectordb