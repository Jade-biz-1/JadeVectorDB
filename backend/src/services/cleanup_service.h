#ifndef JADEVECTORDB_CLEANUP_SERVICE_H
#define JADEVECTORDB_CLEANUP_SERVICE_H

#include "models/vector.h"
#include "models/database.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>

namespace jadevectordb {

// Configuration for cleanup operations
struct CleanupConfig {
    bool enabled = true;                    // Whether cleanup is enabled
    int cleanup_frequency_hours = 24;      // How often to run cleanup (in hours)
    int batch_size = 1000;                 // Number of items to process in each batch
    bool enable_dry_run = false;           // Whether to simulate operations without making changes
    std::string retention_policy_field = "created_at"; // Which field to use for retention
    int default_retention_days = 365;      // Default retention period in days
    bool remove_archived_before_delete = true; // Whether to ensure archived before deletion
    bool log_cleanup_operations = true;    // Whether to log all cleanup operations
    
    CleanupConfig() = default;
};

// Information about a cleanup operation
struct CleanupOperation {
    std::string operation_id;
    std::string database_id;
    std::chrono::system_clock::time_point started_at;
    std::chrono::system_clock::time_point completed_at;
    int items_processed;
    int items_deleted;
    int items_failed;
    std::string status;  // "running", "completed", "failed", "cancelled"
    std::string details;
    
    CleanupOperation() : items_processed(0), items_deleted(0), items_failed(0) {}
    CleanupOperation(const std::string& op_id, const std::string& db_id)
        : operation_id(op_id), database_id(db_id), 
          started_at(std::chrono::system_clock::now()), items_processed(0),
          items_deleted(0), items_failed(0), status("running") {}
};

/**
 * @brief Service to handle automatic cleanup of data that has exceeded retention periods
 * 
 * This service manages the deletion of vector data that exceeds configured retention policies,
 * ensuring compliance with data governance requirements.
 */
class CleanupService {
private:
    std::shared_ptr<logging::Logger> logger_;
    CleanupConfig config_;
    std::unordered_map<std::string, CleanupOperation> active_operations_;  // operation_id -> operation
    std::vector<CleanupOperation> operation_history_;
    std::mutex cleanup_mutex_;
    
public:
    explicit CleanupService();
    ~CleanupService() = default;
    
    // Initialize the cleanup service with configuration
    bool initialize(const CleanupConfig& config);
    
    // Perform cleanup for a specific database based on retention policy
    Result<bool> cleanup_database(const std::string& database_id, 
                                int retention_days = -1);  // -1 means use default
    
    // Perform cleanup for all databases
    Result<bool> cleanup_all_databases();
    
    // Get vectors that should be cleaned up based on retention policy
    Result<std::vector<std::string>> get_vectors_for_cleanup(const std::string& database_id,
                                                           int retention_days = -1) const;
    
    // Delete specific vectors permanently
    Result<bool> delete_vectors(const std::string& database_id, 
                              const std::vector<std::string>& vector_ids);
    
    // Start a scheduled cleanup operation
    Result<std::string> start_scheduled_cleanup();
    
    // Cancel an active cleanup operation
    Result<bool> cancel_cleanup_operation(const std::string& operation_id);
    
    // Get status of an active cleanup operation
    Result<CleanupOperation> get_operation_status(const std::string& operation_id) const;
    
    // Get history of cleanup operations
    Result<std::vector<CleanupOperation>> get_operation_history(int limit = 100) const;
    
    // Get cleanup statistics for a database
    Result<std::unordered_map<std::string, int>> get_cleanup_stats(const std::string& database_id) const;
    
    // Update cleanup configuration
    Result<bool> update_config(const CleanupConfig& new_config);
    
    // Get current cleanup configuration
    CleanupConfig get_config() const;
    
    // Perform a dry run of cleanup (don't actually delete, just report)
    Result<std::vector<std::string>> dry_run_cleanup(const std::string& database_id,
                                                   int retention_days = -1);
    
    // Set retention policy for a specific database
    Result<bool> set_database_retention_policy(const std::string& database_id, int retention_days);
    
    // Get retention policy for a database
    Result<int> get_database_retention_policy(const std::string& database_id) const;

private:
    // Internal helper methods
    
    // Check if a vector has exceeded retention period
    Result<bool> has_exceeded_retention(const Vector& vector, int retention_days) const;
    
    // Calculate retention cutoff time
    std::chrono::system_clock::time_point calculate_cutoff_time(int retention_days) const;
    
    // Process a batch of vectors for cleanup
    Result<bool> process_cleanup_batch(const std::string& database_id,
                                     const std::vector<std::string>& vector_ids,
                                     bool dry_run = false);
    
    // Update operation status in progress tracking
    void update_operation_status(const std::string& operation_id, 
                               const std::string& status,
                               int processed = 0,
                               int deleted = 0,
                               int failed = 0,
                               const std::string& details = "");
    
    // Log cleanup operation for audit trail
    void log_cleanup_operation(const CleanupOperation& operation);
    
    // Validate cleanup configuration
    bool validate_config(const CleanupConfig& config) const;
    
    // Format timestamp for comparison against retention policy
    std::chrono::system_clock::time_point get_timestamp_from_vector(const Vector& vector) const;
    
    // Estimate cleanup time for a database
    Result<int> estimate_cleanup_time(const std::string& database_id, int retention_days) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_CLEANUP_SERVICE_H