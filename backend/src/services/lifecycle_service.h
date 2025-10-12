#ifndef JADEVECTORDB_LIFECYCLE_SERVICE_H
#define JADEVECTORDB_LIFECYCLE_SERVICE_H

#include "models/database.h"
#include "models/vector.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>

namespace jadevectordb {

// Policy for data retention
struct RetentionPolicy {
    int max_age_days = 365;           // Maximum age before deletion (0 = no limit)
    bool archive_on_expire = true;    // Whether to archive before deleting
    int archive_threshold_days = 180; // Age when data gets archived
    bool enable_cleanup = true;       // Whether automatic cleanup is enabled
    std::string cleanup_schedule = "daily"; // How often to run cleanup ("daily", "weekly", "monthly")
    
    RetentionPolicy() = default;
    RetentionPolicy(int max_age, bool archive, int archive_threshold)
        : max_age_days(max_age), archive_on_expire(archive), archive_threshold_days(archive_threshold) {}
};

// Configuration for lifecycle management
struct LifecycleConfig {
    RetentionPolicy retention_policy;
    std::string database_id;
    bool enabled = true;              // Whether lifecycle management is enabled for this DB
    
    LifecycleConfig() = default;
    explicit LifecycleConfig(const std::string& db_id) : database_id(db_id) {}
};

// Information about lifecycle operations
struct LifecycleOperation {
    std::string operation_id;
    std::string database_id;
    std::string vector_id;
    std::string operation_type;       // "archive", "cleanup", "delete"
    std::chrono::system_clock::time_point timestamp;
    bool successful;
    std::string details;
    
    LifecycleOperation() : successful(false) {}
    LifecycleOperation(const std::string& op_id, const std::string& db_id, 
                      const std::string& vec_id, const std::string& op_type)
        : operation_id(op_id), database_id(db_id), vector_id(vec_id), 
          operation_type(op_type), timestamp(std::chrono::system_clock::now()), successful(false) {}
};

/**
 * @brief Service to manage data retention policies and lifecycle operations
 * 
 * This service handles data archival, cleanup, and retention policies for
 * vector databases, ensuring compliance with storage and data governance requirements.
 */
class LifecycleService {
private:
    std::shared_ptr<logging::Logger> logger_;
    std::unordered_map<std::string, LifecycleConfig> config_map_; // database_id -> config
    std::vector<LifecycleOperation> operation_log_;
    std::mutex config_mutex_;
    std::mutex log_mutex_;
    
public:
    explicit LifecycleService();
    ~LifecycleService() = default;
    
    // Initialize the service
    bool initialize();
    
    // Configure retention policy for a database
    Result<bool> configure_retention_policy(const LifecycleConfig& config);
    
    // Get current retention policy for a database
    Result<RetentionPolicy> get_retention_policy(const std::string& database_id) const;
    
    // Check and perform lifecycle operations based on retention policies
    Result<bool> perform_lifecycle_operations();
    
    // Archive vectors that meet archival criteria
    Result<bool> archive_vectors(const std::string& database_id);
    
    // Clean up expired vectors based on retention policy
    Result<bool> cleanup_expired_vectors(const std::string& database_id);
    
    // Delete archived data that has exceeded retention period
    Result<bool> delete_expired_archives(const std::string& database_id);
    
    // Check if a vector has expired based on the retention policy
    Result<bool> is_expired(const std::string& database_id, const Vector& vector) const;
    
    // Check if a vector should be archived based on the retention policy
    Result<bool> should_archive(const std::string& database_id, const Vector& vector) const;
    
    // Manually trigger lifecycle operations for a specific database
    Result<bool> trigger_lifecycle_for_database(const std::string& database_id);
    
    // Get lifecycle operation statistics for a database
    Result<std::unordered_map<std::string, int>> get_lifecycle_stats(const std::string& database_id) const;
    
    // Get operation history for a database
    Result<std::vector<LifecycleOperation>> get_operation_history(const std::string& database_id, 
                                                                int limit = 100) const;
    
    // Log a lifecycle operation
    void log_operation(const LifecycleOperation& operation);
    
    // Get all configured databases
    std::vector<std::string> get_configured_databases() const;
    
    // Update retention policy for a database
    Result<bool> update_retention_policy(const std::string& database_id, 
                                       const RetentionPolicy& new_policy);
    
    // Restore archived data
    Result<bool> restore_archived_data(const std::string& database_id, 
                                     const std::string& archive_id);

private:
    // Internal helper methods
    
    // Check if current time is appropriate for running scheduled operations
    bool should_run_scheduled_operations() const;
    
    // Get vectors that match archival criteria for a database
    Result<std::vector<std::string>> get_vectors_for_archival(const std::string& database_id) const;
    
    // Get vectors that match cleanup criteria for a database
    Result<std::vector<std::string>> get_vectors_for_cleanup(const std::string& database_id) const;
    
    // Archive a specific vector
    Result<bool> archive_vector(const std::string& database_id, const std::string& vector_id);
    
    // Delete a specific vector permanently
    Result<bool> delete_vector_permanently(const std::string& database_id, const std::string& vector_id);
    
    // Calculate age of a vector in days
    int calculate_age_days(const Vector& vector) const;
    
    // Validate retention policy configuration
    bool validate_policy(const RetentionPolicy& policy) const;
    
    // Create archive entry for a vector
    Result<std::string> create_archive_entry(const Vector& vector);
    
    // Schedule next lifecycle operation based on configuration
    void schedule_next_operation();
    
    // Process a single vector through lifecycle management
    Result<bool> process_vector_lifecycle(const std::string& database_id, const Vector& vector);
};

} // namespace jadevectordb

#endif // JADEVECTORDB_LIFECYCLE_SERVICE_H