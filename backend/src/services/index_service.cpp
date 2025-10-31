#include "index_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>

namespace jadevectordb {

IndexService::IndexService() {
    logger_ = logging::LoggerManager::get_logger("IndexService");
}

Result<std::string> IndexService::create_index(const Database& database, const Index& index_config) {
    try {
        std::lock_guard<std::mutex> lock(index_mutex_);
        
        // Generate a unique index ID
        auto now = std::chrono::high_resolution_clock::now();
        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        std::string index_id = "idx_" + database.databaseId + "_" + std::to_string(nanoseconds);
        
        // Store the index
        database_indexes_[database.databaseId][index_id] = index_config;
        
        LOG_INFO(logger_, "Created index " + index_id + " for database " + database.databaseId);
        
        return index_id;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in create_index: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to create index: " + std::string(e.what()));
    }
}

Result<Index> IndexService::get_index(const std::string& database_id, const std::string& index_id) {
    try {
        std::lock_guard<std::mutex> lock(index_mutex_);
        
        auto db_it = database_indexes_.find(database_id);
        if (db_it == database_indexes_.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Database not found: " + database_id);
        }
        
        auto& indexes = db_it->second;
        auto index_it = indexes.find(index_id);
        if (index_it == indexes.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Index not found: " + index_id);
        }
        
        LOG_DEBUG(logger_, "Retrieved index " + index_id + " from database " + database_id);
        return index_it->second;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_index: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get index: " + std::string(e.what()));
    }
}

Result<std::vector<Index>> IndexService::list_indexes(const std::string& database_id) {
    try {
        std::lock_guard<std::mutex> lock(index_mutex_);
        
        auto db_it = database_indexes_.find(database_id);
        if (db_it == database_indexes_.end()) {
            return std::vector<Index>{}; // Return empty vector if database has no indexes
        }
        
        std::vector<Index> indexes;
        for (const auto& pair : db_it->second) {
            indexes.push_back(pair.second);
        }
        
        LOG_DEBUG(logger_, "Listed " + std::to_string(indexes.size()) + " indexes for database " + database_id);
        return indexes;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in list_indexes: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to list indexes: " + std::string(e.what()));
    }
}

Result<void> IndexService::update_index(const std::string& database_id, const std::string& index_id, const Index& new_config) {
    try {
        std::lock_guard<std::mutex> lock(index_mutex_);
        
        auto db_it = database_indexes_.find(database_id);
        if (db_it == database_indexes_.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Database not found: " + database_id);
        }
        
        auto& indexes = db_it->second;
        auto index_it = indexes.find(index_id);
        if (index_it == indexes.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Index not found: " + index_id);
        }
        
        // Update the index configuration
        index_it->second = new_config;
        
        LOG_INFO(logger_, "Updated index " + index_id + " in database " + database_id);
        return Result<void>{};
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_index: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update index: " + std::string(e.what()));
    }
}

Result<void> IndexService::delete_index(const std::string& database_id, const std::string& index_id) {
    try {
        std::lock_guard<std::mutex> lock(index_mutex_);
        
        auto db_it = database_indexes_.find(database_id);
        if (db_it == database_indexes_.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Database not found: " + database_id);
        }
        
        auto& indexes = db_it->second;
        auto index_it = indexes.find(index_id);
        if (index_it == indexes.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Index not found: " + index_id);
        }
        
        // Remove the index
        indexes.erase(index_it);
        
        LOG_INFO(logger_, "Deleted index " + index_id + " from database " + database_id);
        return Result<void>{};
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in delete_index: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to delete index: " + std::string(e.what()));
    }
}

Result<void> IndexService::build_index(const std::string& database_id, const std::string& index_id) {
    try {
        std::lock_guard<std::mutex> lock(index_mutex_);
        
        auto db_it = database_indexes_.find(database_id);
        if (db_it == database_indexes_.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Database not found: " + database_id);
        }
        
        auto& indexes = db_it->second;
        auto index_it = indexes.find(index_id);
        if (index_it == indexes.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Index not found: " + index_id);
        }
        
        // Mark the index as being built
        index_it->second.status = "building";
        
        // In a real implementation, this would trigger the actual index building process
        // For now, we'll mark it as ready after "building"
        index_it->second.status = "ready";
        index_it->second.last_modified = std::chrono::system_clock::now();
        
        LOG_INFO(logger_, "Built index " + index_id + " for database " + database_id);
        return Result<void>{};
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in build_index: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to build index: " + std::string(e.what()));
    }
}

Result<bool> IndexService::is_index_ready(const std::string& database_id, const std::string& index_id) {
    try {
        std::lock_guard<std::mutex> lock(index_mutex_);
        
        auto db_it = database_indexes_.find(database_id);
        if (db_it == database_indexes_.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Database not found: " + database_id);
        }
        
        auto& indexes = db_it->second;
        auto index_it = indexes.find(index_id);
        if (index_it == indexes.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Index not found: " + index_id);
        }
        
        bool is_ready = (index_it->second.status == "ready");
        
        LOG_DEBUG(logger_, "Index " + index_id + " in database " + database_id + 
                 " is " + (is_ready ? "" : "not ") + "ready");
        return is_ready;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in is_index_ready: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check index readiness: " + std::string(e.what()));
    }
}

Result<std::map<std::string, double>> IndexService::get_index_stats(const std::string& database_id, const std::string& index_id) {
    try {
        std::lock_guard<std::mutex> lock(index_mutex_);
        
        auto db_it = database_indexes_.find(database_id);
        if (db_it == database_indexes_.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Database not found: " + database_id);
        }
        
        auto& indexes = db_it->second;
        auto index_it = indexes.find(index_id);
        if (index_it == indexes.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Index not found: " + index_id);
        }
        
        std::map<std::string, double> stats;
        stats["dimensionality"] = static_cast<double>(index_it->second.dimensions);
        stats["vector_count"] = static_cast<double>(index_it->second.vector_count);
        stats["size_bytes"] = static_cast<double>(index_it->second.size_bytes);
        stats["memory_usage_mb"] = static_cast<double>(index_it->second.size_bytes) / (1024 * 1024);
        
        LOG_DEBUG(logger_, "Retrieved stats for index " + index_id + " in database " + database_id);
        return stats;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_index_stats: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get index stats: " + std::string(e.what()));
    }
}

} // namespace jadevectordb