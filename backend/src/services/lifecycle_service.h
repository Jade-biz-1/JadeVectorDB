#ifndef JADEVECTORDB_LIFECYCLE_SERVICE_H
#define JADEVECTORDB_LIFECYCLE_SERVICE_H

#include <string>
#include <memory>
#include <vector>
#include <chrono>

#include "lib/error_handling.h"
#include "models/database.h"

namespace jadevectordb {

// Lifecycle management service
class LifecycleService {
private:
    std::shared_ptr<logging::Logger> logger_;
    
    // Policy configuration for each database
    std::unordered_map<std::string, std::pair<std::chrono::hours, bool>> retention_policies_;
    mutable std::mutex policy_mutex_;
    
    // Scheduling information
    std::unordered_map<std::string, std::chrono::system_clock::time_point> next_cleanup_times_;
    mutable std::mutex schedule_mutex_;

public:
    LifecycleService();
    ~LifecycleService() = default;
    
    // Configure retention policy for a database
    Result<void> configure_retention_policy(const std::string& database_id, 
                                           const std::chrono::hours& max_age,
                                           bool archive_on_expire = false);
    
    // Get current retention policy for a database
    Result<std::pair<std::chrono::hours, bool>> get_retention_policy(const std::string& database_id);
    
    // Perform cleanup operations on expired data
    Result<void> perform_cleanup(const std::string& database_id);
    
    // Perform archival operations on older data
    Result<void> perform_archival(const std::string& database_id);
    
    // Get lifecycle status for a database
    Result<std::string> get_lifecycle_status(const std::string& database_id);
    
    // Initialize lifecycle management for a database
    Result<void> initialize_lifecycle_for_database(const Database& database);
    
    // Get cleanup schedule
    Result<std::chrono::system_clock::time_point> get_next_cleanup_time(const std::string& database_id);
};

} // namespace jadevectordb

#endif // JADEVECTORDB_LIFECYCLE_SERVICE_H