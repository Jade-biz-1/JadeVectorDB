#include "cleanup_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <thread>
#include <chrono>

namespace jadevectordb {

CleanupService::CleanupService() {
    logger_ = logging::LoggerManager::get_logger("CleanupService");
}

bool CleanupService::initialize(const CleanupConfig& config) {
    try {
        std::lock_guard<std::mutex> lock(cleanup_mutex_);
        
        if (!validate_config(config)) {
            LOG_ERROR(logger_, "Invalid cleanup configuration provided");
            return false;
        }
        
        config_ = config;
        
        LOG_INFO(logger_, "CleanupService initialized with frequency: " + 
                std::to_string(config_.cleanup_frequency_hours) + " hours, " +
                "batch size: " + std::to_string(config_.batch_size));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in CleanupService::initialize: " + std::string(e.what()));
        return false;
    }
}

Result<bool> CleanupService::cleanup_database(const std::string& database_id, int retention_days) {
    try {
        LOG_INFO(logger_, "Starting cleanup for database: " + database_id);
        
        // Use default retention if not specified
        int actual_retention_days = (retention_days == -1) ? config_.default_retention_days : retention_days;
        
        // Generate operation ID
        std::string operation_id = "cleanup_" + database_id + "_" + 
                                  std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        
        // Add operation to active operations
        {
            std::lock_guard<std::mutex> lock(cleanup_mutex_);
            active_operations_[operation_id] = CleanupOperation(operation_id, database_id);
        }
        
        LOG_DEBUG(logger_, "Operation " + operation_id + " started for database " + database_id);
        
        // Get vectors for cleanup
        auto vectors_result = get_vectors_for_cleanup(database_id, actual_retention_days);
        if (!vectors_result.has_value()) {
            update_operation_status(operation_id, "failed", 0, 0, 1, 
                                  "Failed to get vectors for cleanup: " + 
                                  ErrorHandler::format_error(vectors_result.error()));
            return vectors_result;
        }
        
        auto vectors_to_delete = vectors_result.value();
        
        if (config_.log_cleanup_operations) {
            LOG_DEBUG(logger_, "Found " + std::to_string(vectors_to_delete.size()) + 
                     " vectors for cleanup in database " + database_id);
        }
        
        // Process vectors in batches
        int items_processed = 0;
        int items_deleted = 0;
        int items_failed = 0;
        
        for (size_t i = 0; i < vectors_to_delete.size(); i += static_cast<size_t>(config_.batch_size)) {
            size_t batch_end = std::min(i + static_cast<size_t>(config_.batch_size), vectors_to_delete.size());
            std::vector<std::string> batch(vectors_to_delete.begin() + i, vectors_to_delete.begin() + batch_end);
            
            // Process this batch
            auto batch_result = process_cleanup_batch(database_id, batch, config_.enable_dry_run);
            if (!batch_result.has_value()) {
                items_failed += static_cast<int>(batch.size());
                LOG_WARN(logger_, "Failed to process cleanup batch: " + 
                        ErrorHandler::format_error(batch_result.error()));
            } else {
                if (!config_.enable_dry_run) {
                    items_deleted += static_cast<int>(batch.size());
                } else {
                    items_deleted += 0; // In dry run, we don't actually delete
                }
            }
            
            items_processed += static_cast<int>(batch.size());
            
            // Update operation status
            update_operation_status(operation_id, "running", items_processed, items_deleted, items_failed);
            
            // Small delay between batches to avoid overwhelming the system
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Update final operation status
        {
            std::lock_guard<std::mutex> lock(cleanup_mutex_);
            auto& operation = active_operations_[operation_id];
            operation.items_processed = items_processed;
            operation.items_deleted = items_deleted;
            operation.items_failed = items_failed;
            operation.status = "completed";
            operation.completed_at = std::chrono::system_clock::now();
            
            // Move operation to history
            operation_history_.push_back(operation);
            active_operations_.erase(operation_id);
        }
        
        LOG_INFO(logger_, "Cleanup completed for database " + database_id + 
                ". Processed: " + std::to_string(items_processed) + 
                ", Deleted: " + std::to_string(items_deleted) + 
                ", Failed: " + std::to_string(items_failed));
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in cleanup_database: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to cleanup database: " + std::string(e.what()));
    }
}

Result<bool> CleanupService::cleanup_all_databases() {
    try {
        LOG_INFO(logger_, "Starting cleanup for all databases");
        
        // In a real implementation, we would get a list of all databases
        // For now, we'll simulate with a list of database IDs
        std::vector<std::string> all_databases = {"db1", "db2", "db3"}; // This should come from database service
        
        bool all_success = true;
        for (const auto& database_id : all_databases) {
            auto result = cleanup_database(database_id);
            if (!result.has_value()) {
                LOG_WARN(logger_, "Failed to cleanup database " + database_id + ": " + 
                        ErrorHandler::format_error(result.error()));
                all_success = false;
            }
        }
        
        LOG_INFO(logger_, "Cleanup completed for all databases");
        return all_success;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in cleanup_all_databases: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to cleanup all databases: " + std::string(e.what()));
    }
}

Result<std::vector<std::string>> CleanupService::get_vectors_for_cleanup(const std::string& database_id,
                                                                       int retention_days) const {
    try {
        LOG_DEBUG(logger_, "Getting vectors for cleanup in database: " + database_id);
        
        // Use default retention if not specified
        int actual_retention_days = (retention_days == -1) ? config_.default_retention_days : retention_days;
        
        // In a real implementation, this would query the database for vectors that exceed retention
        // For now, we'll return an empty list since we don't have access to the actual database
        std::vector<std::string> vectors_to_cleanup;
        
        // Calculate cutoff time
        auto cutoff_time = calculate_cutoff_time(actual_retention_days);
        
        // For demonstration, we'll return a few dummy vectors
        // In a real implementation, you would connect to the database service
        // and fetch vectors that were created before the cutoff time
        if (database_id == "db1") {
            vectors_to_cleanup = {"vector1", "vector2", "vector3"};
        } else if (database_id == "db2") {
            vectors_to_cleanup = {"vector4", "vector5"};
        }
        
        LOG_DEBUG(logger_, "Found " + std::to_string(vectors_to_cleanup.size()) + 
                 " vectors for cleanup in database: " + database_id);
        return vectors_to_cleanup;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_vectors_for_cleanup: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get vectors for cleanup: " + std::string(e.what()));
    }
}

Result<bool> CleanupService::delete_vectors(const std::string& database_id, 
                                          const std::vector<std::string>& vector_ids) {
    try {
        LOG_DEBUG(logger_, "Deleting " + std::to_string(vector_ids.size()) + " vectors from database: " + database_id);
        
        // In a real implementation, this would delete the vectors from the database
        // For now, we'll just return success and log the operation
        for (const auto& vector_id : vector_ids) {
            LOG_DEBUG(logger_, "Deleting vector: " + vector_id + " from database: " + database_id);
        }
        
        if (config_.log_cleanup_operations) {
            log_cleanup_operation(CleanupOperation("delete_op_" + database_id, database_id));
        }
        
        LOG_DEBUG(logger_, "Successfully deleted " + std::to_string(vector_ids.size()) + 
                 " vectors from database: " + database_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in delete_vectors: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to delete vectors: " + std::string(e.what()));
    }
}

Result<std::string> CleanupService::start_scheduled_cleanup() {
    try {
        LOG_INFO(logger_, "Starting scheduled cleanup");
        
        std::string operation_id = "scheduled_cleanup_" + 
                                  std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        
        // This would typically run as a background task
        // For now, we'll just start it in the current thread
        std::thread cleanup_thread([this, operation_id]() {
            // Add operation to active operations
            {
                std::lock_guard<std::mutex> lock(cleanup_mutex_);
                active_operations_[operation_id] = CleanupOperation(operation_id, "all_databases");
            }
            
            // Perform cleanup for all databases
            auto result = cleanup_all_databases();
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Scheduled cleanup failed: " + ErrorHandler::format_error(result.error()));
                update_operation_status(operation_id, "failed", 0, 0, 1, 
                                      "Scheduled cleanup failed: " + ErrorHandler::format_error(result.error()));
            } else {
                update_operation_status(operation_id, "completed", 0, 0, 0, "Scheduled cleanup completed successfully");
            }
        });
        
        cleanup_thread.detach(); // Detach since we don't want to block
        
        LOG_INFO(logger_, "Scheduled cleanup started with ID: " + operation_id);
        return operation_id;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in start_scheduled_cleanup: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to start scheduled cleanup: " + std::string(e.what()));
    }
}

Result<bool> CleanupService::cancel_cleanup_operation(const std::string& operation_id) {
    try {
        std::lock_guard<std::mutex> lock(cleanup_mutex_);
        
        auto it = active_operations_.find(operation_id);
        if (it == active_operations_.end()) {
            RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Operation not found: " + operation_id);
        }
        
        // Update operation status to cancelled
        it->second.status = "cancelled";
        it->second.completed_at = std::chrono::system_clock::now();
        
        // Move to operation history
        operation_history_.push_back(it->second);
        active_operations_.erase(it);
        
        LOG_INFO(logger_, "Cancelled cleanup operation: " + operation_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in cancel_cleanup_operation: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to cancel operation: " + std::string(e.what()));
    }
}

Result<CleanupOperation> CleanupService::get_operation_status(const std::string& operation_id) const {
    try {
        std::lock_guard<std::mutex> lock(cleanup_mutex_);
        
        auto it = active_operations_.find(operation_id);
        if (it != active_operations_.end()) {
            return it->second;
        }
        
        // Check operation history as well
        for (const auto& operation : operation_history_) {
            if (operation.operation_id == operation_id) {
                return operation;
            }
        }
        
        RETURN_ERROR(ErrorCode::RESOURCE_NOT_FOUND, "Operation not found: " + operation_id);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_operation_status: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get operation status: " + std::string(e.what()));
    }
}

Result<std::vector<CleanupOperation>> CleanupService::get_operation_history(int limit) const {
    try {
        std::lock_guard<std::mutex> lock(cleanup_mutex_);
        
        // Return the most recent operations up to the limit
        size_t start_idx = operation_history_.size() > static_cast<size_t>(limit) ? 
                          operation_history_.size() - limit : 0;
        
        std::vector<CleanupOperation> result;
        for (size_t i = start_idx; i < operation_history_.size(); ++i) {
            result.push_back(operation_history_[i]);
        }
        
        // Return in reverse order (most recent first)
        std::reverse(result.begin(), result.end());
        
        LOG_DEBUG(logger_, "Retrieved " + std::to_string(result.size()) + " operations from history");
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_operation_history: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get operation history: " + std::string(e.what()));
    }
}

Result<std::unordered_map<std::string, int>> CleanupService::get_cleanup_stats(const std::string& database_id) const {
    try {
        std::lock_guard<std::mutex> lock(cleanup_mutex_);
        
        std::unordered_map<std::string, int> stats;
        stats["total_active_operations"] = static_cast<int>(active_operations_.size());
        stats["total_history_operations"] = static_cast<int>(operation_history_.size());
        
        // Count operations by status
        int completed = 0, running = 0, failed = 0, cancelled = 0;
        for (const auto& op : operation_history_) {
            if (op.database_id == database_id || database_id == "all_databases") {
                if (op.status == "completed") completed++;
                else if (op.status == "running") running++;
                else if (op.status == "failed") failed++;
                else if (op.status == "cancelled") cancelled++;
            }
        }
        
        stats["completed_operations"] = completed;
        stats["failed_operations"] = failed;
        stats["cancelled_operations"] = cancelled;
        
        LOG_DEBUG(logger_, "Generated cleanup statistics for database: " + database_id);
        return stats;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_cleanup_stats: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get cleanup stats: " + std::string(e.what()));
    }
}

Result<bool> CleanupService::update_config(const CleanupConfig& new_config) {
    try {
        std::lock_guard<std::mutex> lock(cleanup_mutex_);
        
        if (!validate_config(new_config)) {
            LOG_ERROR(logger_, "Invalid cleanup configuration provided for update");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid cleanup configuration");
        }
        
        config_ = new_config;
        
        LOG_INFO(logger_, "Updated cleanup configuration");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_config: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update cleanup configuration: " + std::string(e.what()));
    }
}

CleanupConfig CleanupService::get_config() const {
    std::lock_guard<std::mutex> lock(cleanup_mutex_);
    return config_;
}

Result<std::vector<std::string>> CleanupService::dry_run_cleanup(const std::string& database_id,
                                                               int retention_days) {
    try {
        LOG_INFO(logger_, "Performing dry run cleanup for database: " + database_id);
        
        // Enable dry run mode temporarily
        bool original_dry_run = config_.enable_dry_run;
        {
            std::lock_guard<std::mutex> lock(cleanup_mutex_);
            config_.enable_dry_run = true;
        }
        
        // Get vectors that would be cleaned up
        auto result = get_vectors_for_cleanup(database_id, retention_days);
        
        // Restore original dry run setting
        {
            std::lock_guard<std::mutex> lock(cleanup_mutex_);
            config_.enable_dry_run = original_dry_run;
        }
        
        if (result.has_value()) {
            LOG_INFO(logger_, "Dry run identified " + std::to_string(result.value().size()) + 
                    " vectors for cleanup in database: " + database_id);
        }
        
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in dry_run_cleanup: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to perform dry run: " + std::string(e.what()));
    }
}

Result<bool> CleanupService::set_database_retention_policy(const std::string& database_id, int retention_days) {
    try {
        LOG_INFO(logger_, "Setting retention policy for database " + database_id + 
                " to " + std::to_string(retention_days) + " days");
        
        // In a real implementation, this would update the database's retention policy in persistent storage
        // For now, we'll just log the operation
        
        if (retention_days < 0) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Retention days must be non-negative");
        }
        
        LOG_INFO(logger_, "Set retention policy for database " + database_id + 
                " to " + std::to_string(retention_days) + " days");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in set_database_retention_policy: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to set retention policy: " + std::string(e.what()));
    }
}

Result<int> CleanupService::get_database_retention_policy(const std::string& database_id) const {
    try {
        LOG_DEBUG(logger_, "Getting retention policy for database: " + database_id);
        
        // In a real implementation, this would retrieve the database's retention policy from persistent storage
        // For now, return the default retention
        
        LOG_DEBUG(logger_, "Retrieved retention policy for database " + database_id + 
                " as " + std::to_string(config_.default_retention_days) + " days");
        return config_.default_retention_days;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_database_retention_policy: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get retention policy: " + std::string(e.what()));
    }
}

// Private methods

Result<bool> CleanupService::has_exceeded_retention(const Vector& vector, int retention_days) const {
    try {
        // Calculate cutoff time
        auto cutoff_time = calculate_cutoff_time(retention_days);
        
        // Get the timestamp from the vector
        auto vector_time = get_timestamp_from_vector(vector);
        
        // Return true if the vector's time is earlier than the cutoff
        bool exceeded = vector_time < cutoff_time;
        
        LOG_DEBUG(logger_, "Vector " + vector.id + " retention check: " + 
                 (exceeded ? "EXCEEDED" : "NOT EXCEEDED") + 
                 " cutoff of " + std::to_string(retention_days) + " days");
        return exceeded;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in has_exceeded_retention: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check retention: " + std::string(e.what()));
    }
}

std::chrono::system_clock::time_point CleanupService::calculate_cutoff_time(int retention_days) const {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::hours(24 * retention_days);
    return now - duration;
}

Result<bool> CleanupService::process_cleanup_batch(const std::string& database_id,
                                                 const std::vector<std::string>& vector_ids,
                                                 bool dry_run) {
    try {
        LOG_DEBUG(logger_, "Processing cleanup batch of " + std::to_string(vector_ids.size()) + 
                 " vectors in database: " + database_id + 
                 (dry_run ? " (dry run)" : ""));
        
        if (!dry_run) {
            // Actually delete the vectors
            auto delete_result = delete_vectors(database_id, vector_ids);
            if (!delete_result.has_value()) {
                LOG_ERROR(logger_, "Failed to delete vectors in batch: " + 
                         ErrorHandler::format_error(delete_result.error()));
                return delete_result;
            }
        } else {
            LOG_DEBUG(logger_, "Dry run: Would have deleted " + std::to_string(vector_ids.size()) + 
                     " vectors from database: " + database_id);
        }
        
        LOG_DEBUG(logger_, "Successfully processed cleanup batch in database: " + database_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in process_cleanup_batch: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to process cleanup batch: " + std::string(e.what()));
    }
}

void CleanupService::update_operation_status(const std::string& operation_id, 
                                           const std::string& status,
                                           int processed,
                                           int deleted,
                                           int failed,
                                           const std::string& details) {
    std::lock_guard<std::mutex> lock(cleanup_mutex_);
    
    auto it = active_operations_.find(operation_id);
    if (it != active_operations_.end()) {
        it->second.status = status;
        it->second.items_processed += processed;
        it->second.items_deleted += deleted;
        it->second.items_failed += failed;
        if (!details.empty()) {
            it->second.details = details;
        }
    }
}

void CleanupService::log_cleanup_operation(const CleanupOperation& operation) {
    if (config_.log_cleanup_operations) {
        LOG_INFO(logger_, "Cleanup Operation - ID: " + operation.operation_id + 
                ", DB: " + operation.database_id + 
                ", Status: " + operation.status + 
                ", Processed: " + std::to_string(operation.items_processed) + 
                ", Deleted: " + std::to_string(operation.items_deleted) + 
                ", Failed: " + std::to_string(operation.items_failed));
    }
}

bool CleanupService::validate_config(const CleanupConfig& config) const {
    // Basic validation
    if (config.cleanup_frequency_hours <= 0) {
        LOG_ERROR(logger_, "Invalid cleanup frequency: " + std::to_string(config.cleanup_frequency_hours));
        return false;
    }
    
    if (config.batch_size <= 0) {
        LOG_ERROR(logger_, "Invalid batch size: " + std::to_string(config.batch_size));
        return false;
    }
    
    if (config.default_retention_days <= 0) {
        LOG_ERROR(logger_, "Invalid default retention days: " + std::to_string(config.default_retention_days));
        return false;
    }
    
    if (config.retention_policy_field.empty()) {
        LOG_ERROR(logger_, "Retention policy field cannot be empty");
        return false;
    }
    
    return true;
}

std::chrono::system_clock::time_point CleanupService::get_timestamp_from_vector(const Vector& vector) const {
    // In a real implementation, this would extract the timestamp from the vector's metadata
    // For now, we'll return the current time as a placeholder
    return std::chrono::system_clock::now();
}

Result<int> CleanupService::estimate_cleanup_time(const std::string& database_id, int retention_days) const {
    try {
        // This is a very rough estimate
        // In a real implementation, this would analyze the database size and structure
        auto vectors_result = get_vectors_for_cleanup(database_id, retention_days);
        if (!vectors_result.has_value()) {
            return vectors_result;
        }
        
        auto vectors = vectors_result.value();
        // Estimate: 1ms per vector to process (very rough estimate)
        int estimated_time_ms = static_cast<int>(vectors.size()) * 1;
        
        LOG_DEBUG(logger_, "Estimated cleanup time for database " + database_id + 
                 " is approximately " + std::to_string(estimated_time_ms) + " ms");
        return estimated_time_ms;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in estimate_cleanup_time: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to estimate cleanup time: " + std::string(e.what()));
    }
}

} // namespace jadevectordb