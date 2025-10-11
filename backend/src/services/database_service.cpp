#include "database_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include "services/database_layer.h" // Assuming this is the database abstraction layer

namespace jadevectordb {

DatabaseService::DatabaseService() {
    logger_ = logging::LoggerManager::get_logger("DatabaseService");
}

Result<std::string> DatabaseService::create_database(const Database& db_config) {
    // Validate database configuration
    if (!db_config.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid database configuration");
    }
    
    // Check if database with this name already exists
    {
        std::shared_lock<std::shared_mutex> lock(databases_mutex_);
        for (const auto& [id, db] : databases_) {
            if (db.name == db_config.name) {
                RETURN_ERROR(ErrorCode::ALREADY_EXISTS, "Database with this name already exists: " + db_config.name);
            }
        }
    }
    
    // In a real implementation, we would store this in persistent storage
    // For now, we'll use the in-memory approach as an example
    std::string database_id = generate_id();
    Database new_db = db_config;
    new_db.databaseId = database_id;
    new_db.created_at = "2025-10-11T00:00:00Z"; // In a real system, use current timestamp
    new_db.updated_at = "2025-10-11T00:00:00Z";
    
    std::unique_lock<std::shared_mutex> lock(databases_mutex_);
    databases_[database_id] = std::make_unique<Database>(new_db);
    
    LOG_INFO(logger_, "Created database: " << database_id << " (" << db_config.name << ")");
    return database_id;
}

Result<Database> DatabaseService::get_database(const std::string& database_id) const {
    std::shared_lock<std::shared_mutex> lock(databases_mutex_);
    
    auto it = databases_.find(database_id);
    if (it == databases_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    return *(it->second);
}

Result<std::vector<Database>> DatabaseService::list_databases() const {
    std::shared_lock<std::shared_mutex> lock(databases_mutex_);
    
    std::vector<Database> result;
    for (const auto& [id, db] : databases_) {
        result.push_back(*(db.get()));
    }
    
    return result;
}

Result<void> DatabaseService::update_database(const std::string& database_id, const Database& new_config) {
    if (!new_config.validate()) {
        RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid database configuration");
    }
    
    std::unique_lock<std::shared_mutex> lock(databases_mutex_);
    
    auto it = databases_.find(database_id);
    if (it == databases_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    // Update the database configuration
    auto updated_db = std::make_unique<Database>(new_config);
    updated_db->databaseId = database_id;
    updated_db->updated_at = "2025-10-11T00:00:00Z"; // In a real system, use current timestamp
    
    databases_[database_id] = std::move(updated_db);
    
    LOG_INFO(logger_, "Updated database: " << database_id);
    return {};
}

Result<void> DatabaseService::delete_database(const std::string& database_id) {
    std::unique_lock<std::shared_mutex> lock(databases_mutex_);
    
    auto it = databases_.find(database_id);
    if (it == databases_.end()) {
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Database not found: " + database_id);
    }
    
    databases_.erase(it);
    
    LOG_INFO(logger_, "Deleted database: " << database_id);
    return {};
}

bool DatabaseService::database_exists(const std::string& database_id) const {
    std::shared_lock<std::shared_mutex> lock(databases_mutex_);
    return databases_.find(database_id) != databases_.end();
}

size_t DatabaseService::get_database_count() const {
    std::shared_lock<std::shared_mutex> lock(databases_mutex_);
    return databases_.size();
}

std::string DatabaseService::generate_id() const {
    // Generate a unique ID
    // In a real implementation, this should use a more robust ID generation method
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto count = duration.count();
    
    std::stringstream ss;
    ss << std::hex << count;
    return ss.str();
}

} // namespace jadevectordb