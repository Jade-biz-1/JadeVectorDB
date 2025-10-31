#include "lifecycle_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <chrono>

namespace jadevectordb {

LifecycleService::LifecycleService() {
    logger_ = logging::LoggerManager::get_logger("LifecycleService");
}

Result<void> LifecycleService::configure_retention_policy(const std::string& database_id, 
                                                         const std::chrono::hours& max_age,
                                                         bool archive_on_expire) {
    try {
        std::lock_guard<std::mutex> lock(policy_mutex_);
        
        retention_policies_[database_id] = std::make_pair(max_age, archive_on_expire);
        
        // Update next cleanup time based on the retention policy
        std::lock_guard<std::mutex> schedule_lock(schedule_mutex_);
        next_cleanup_times_[database_id] = std::chrono::system_clock::now() + max_age;
        
        LOG_INFO(logger_, "Configured retention policy for database " + database_id + 
                " with max age " + std::to_string(max_age.count()) + " hours, " +
                (archive_on_expire ? "with" : "without") + " archival on expire");
        return Result<void>{};
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in configure_retention_policy: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to configure retention policy: " + std::string(e.what()));
    }
}

Result<std::pair<std::chrono::hours, bool>> LifecycleService::get_retention_policy(const std::string& database_id) {
    try {
        std::lock_guard<std::mutex> lock(policy_mutex_);
        
        auto it = retention_policies_.find(database_id);
        if (it != retention_policies_.end()) {
            LOG_DEBUG(logger_, "Retrieved retention policy for database " + database_id);
            return it->second;
        }
        
        // Return default policy if not found
        LOG_DEBUG(logger_, "No specific retention policy found for database " + database_id + 
                 ", returning default: 30 days");
        return std::make_pair(std::chrono::hours(24 * 30), false);  // Default: 30 days, no archiving
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_retention_policy: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get retention policy: " + std::string(e.what()));
    }
}

Result<void> LifecycleService::perform_cleanup(const std::string& database_id) {
    try {
        LOG_INFO(logger_, "Performing cleanup for database: " + database_id);
        
        auto policy_result = get_retention_policy(database_id);
        if (!policy_result.has_value()) {
            LOG_WARN(logger_, "Could not get retention policy for database " + database_id);
            return policy_result;
        }
        
        auto [max_age, archive_on_expire] = policy_result.value();
        
        if (archive_on_expire) {
            auto archival_result = perform_archival(database_id);
            if (!archival_result.has_value()) {
                LOG_WARN(logger_, "Archival failed during cleanup for database " + database_id + 
                        ", continuing with deletion: " + ErrorHandler::format_error(archival_result.error()));
            }
        }
        
        // In a real implementation, this would connect to the database service
        // and remove data that exceeds the retention policy
        // For now, we'll just log the operation
        
        LOG_INFO(logger_, "Cleanup completed for database: " + database_id);
        return Result<void>{};
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in perform_cleanup: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to perform cleanup: " + std::string(e.what()));
    }
}

Result<void> LifecycleService::perform_archival(const std::string& database_id) {
    try {
        LOG_INFO(logger_, "Performing archival for database: " + database_id);
        
        // In a real implementation, this would connect to the archival service
        // and move data that's approaching its retention limit to archival storage
        // For now, we'll just log the operation
        
        LOG_INFO(logger_, "Archival completed for database: " + database_id);
        return Result<void>{};
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in perform_archival: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to perform archival: " + std::string(e.what()));
    }
}

Result<std::string> LifecycleService::get_lifecycle_status(const std::string& database_id) {
    try {
        // Check if database has any lifecycle operations scheduled or in progress
        auto policy_result = get_retention_policy(database_id);
        if (!policy_result.has_value()) {
            LOG_WARN(logger_, "Could not get retention policy for database " + database_id);
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get retention policy: " + std::string("No policy found"));
        }
        
        // Determine status based on next cleanup time
        std::lock_guard<std::mutex> schedule_lock(schedule_mutex_);
        auto next_cleanup_it = next_cleanup_times_.find(database_id);
        if (next_cleanup_it != next_cleanup_times_.end()) {
            if (next_cleanup_it->second <= std::chrono::system_clock::now()) {
                return std::string("pending_cleanup");
            }
        }
        
        LOG_DEBUG(logger_, "Retrieved lifecycle status for database " + database_id);
        return std::string("active");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_lifecycle_status: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get lifecycle status: " + std::string(e.what()));
    }
}

Result<void> LifecycleService::initialize_lifecycle_for_database(const Database& database) {
    try {
        LOG_INFO(logger_, "Initializing lifecycle management for database: " + database.databaseId);
        
        // Set up default lifecycle policies for the new database
        Result<void> result = configure_retention_policy(database.databaseId, 
                                                         std::chrono::hours(24 * 30),  // 30 days default
                                                         false);  // No archival by default
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to initialize lifecycle for database " + database.databaseId);
            return result;
        }
        
        LOG_INFO(logger_, "Lifecycle management initialized for database: " + database.databaseId);
        return Result<void>{};
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in initialize_lifecycle_for_database: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to initialize lifecycle: " + std::string(e.what()));
    }
}

Result<std::chrono::system_clock::time_point> LifecycleService::get_next_cleanup_time(const std::string& database_id) {
    try {
        std::lock_guard<std::mutex> lock(schedule_mutex_);
        
        auto it = next_cleanup_times_.find(database_id);
        if (it != next_cleanup_times_.end()) {
            LOG_DEBUG(logger_, "Retrieved next cleanup time for database " + database_id);
            return it->second;
        }
        
        // Return default if not found (24 hours from now)
        auto default_time = std::chrono::system_clock::now() + std::chrono::hours(24);
        LOG_DEBUG(logger_, "No scheduled cleanup found for database " + database_id + 
                 ", returning default time");
        return default_time;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_next_cleanup_time: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get next cleanup time: " + std::string(e.what()));
    }
}

} // namespace jadevectordb