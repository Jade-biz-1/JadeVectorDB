#ifndef JADEVECTORDB_SECURITY_AUDIT_LOGGER_H
#define JADEVECTORDB_SECURITY_AUDIT_LOGGER_H

#include "lib/logging.h"
#include "lib/error_handling.h"
#include "models/vector.h"
#include "models/database.h"
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <fstream>

namespace jadevectordb {

// Types of security events to log
enum class SecurityEventType {
    AUTHENTICATION_ATTEMPT,
    AUTHENTICATION_SUCCESS,
    AUTHENTICATION_FAILURE,
    AUTHORIZATION_CHECK,
    AUTHORIZATION_GRANTED,
    AUTHORIZATION_DENIED,
    DATA_ACCESS,
    DATA_MODIFICATION,
    DATA_DELETION,
    CONFIGURATION_CHANGE,
    ADMIN_OPERATION,
    SECURITY_POLICY_VIOLATION
};

// Information about a security event
struct SecurityEvent {
    std::string event_id;
    SecurityEventType event_type;
    std::string user_id;
    std::string ip_address;
    std::string resource_accessed;  // Database, vector ID, etc.
    std::string operation;
    std::chrono::system_clock::time_point timestamp;
    std::string details;  // Additional context about the event
    bool success;         // Whether the operation was successful
    std::string session_id;  // Session identifier
    std::string client_info; // Information about the client making the request
    
    SecurityEvent() : success(false) {}
    SecurityEvent(SecurityEventType type, const std::string& user, const std::string& ip, 
                  const std::string& resource, const std::string& op, bool op_success = true)
        : event_type(type), user_id(user), ip_address(ip), 
          resource_accessed(resource), operation(op), 
          timestamp(std::chrono::system_clock::now()), 
          success(op_success) {
        // Generate a unique event ID
        auto now = std::chrono::high_resolution_clock::now();
        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        event_id = "sec_evt_" + std::to_string(nanoseconds);
    }
};

// Configuration for the security audit logger
struct SecurityAuditConfig {
    bool enabled = true;                        // Whether audit logging is enabled
    std::string log_file_path = "./security_audit.log";  // Path to the audit log file
    int max_log_file_size_mb = 100;            // Maximum size of log file before rotation
    int max_log_files = 5;                     // Number of rotated log files to keep
    std::vector<SecurityEventType> events_to_log; // Which events to log
    bool log_all_operations = false;           // Whether to log all operations regardless of type
    bool include_request_details = true;       // Whether to include details about requests
    std::string log_format = "json";           // Format of log entries: "json", "csv", etc.
    
    SecurityAuditConfig() {
        // By default, log the most critical security events
        events_to_log = {
            SecurityEventType::AUTHENTICATION_ATTEMPT,
            SecurityEventType::AUTHENTICATION_FAILURE,
            SecurityEventType::AUTHORIZATION_DENIED,
            SecurityEventType::DATA_DELETION,
            SecurityEventType::CONFIGURATION_CHANGE,
            SecurityEventType::SECURITY_POLICY_VIOLATION
        };
    }
};

/**
 * @brief Security Audit Logger for tracking security-related events
 * 
 * This service records all security-related events in the system including
 * authentication attempts, authorization decisions, data access, and other
 * security-relevant activities to maintain an audit trail.
 */
class SecurityAuditLogger {
private:
    std::shared_ptr<logging::Logger> logger_;
    SecurityAuditConfig config_;
    std::ofstream log_file_;
    mutable std::mutex log_mutex_;
    size_t current_log_size_;
    
public:
    explicit SecurityAuditLogger();
    ~SecurityAuditLogger();
    
    // Initialize the security audit logger with configuration
    bool initialize(const SecurityAuditConfig& config);
    
    // Log a security event
    Result<bool> log_security_event(const SecurityEvent& event);
    
    // Log authentication attempt
    Result<bool> log_authentication_attempt(const std::string& user_id, 
                                          const std::string& ip_address,
                                          bool success,
                                          const std::string& details = "");
    
    // Log authorization check
    Result<bool> log_authorization_check(const std::string& user_id,
                                       const std::string& resource,
                                       const std::string& operation,
                                       bool granted,
                                       const std::string& details = "");
    
    // Log data access
    Result<bool> log_data_access(const std::string& user_id,
                               const std::string& ip_address,
                               const std::string& database_id,
                               const std::string& vector_id,
                               bool success,
                               const std::string& details = "");
    
    // Log data modification
    Result<bool> log_data_modification(const std::string& user_id,
                                     const std::string& ip_address,
                                     const std::string& database_id,
                                     const std::string& vector_id,
                                     bool success,
                                     const std::string& details = "");
    
    // Log data deletion
    Result<bool> log_data_deletion(const std::string& user_id,
                                 const std::string& ip_address,
                                 const std::string& database_id,
                                 const std::string& vector_id,
                                 bool success,
                                 const std::string& details = "");
    
    // Log a configuration change
    Result<bool> log_configuration_change(const std::string& user_id,
                                        const std::string& ip_address,
                                        const std::string& config_change,
                                        bool success,
                                        const std::string& details = "");
    
    // Bulk log multiple events
    Result<bool> log_security_events(const std::vector<SecurityEvent>& events);
    
    // Update audit logging configuration
    Result<bool> update_config(const SecurityAuditConfig& new_config);
    
    // Get current audit logging configuration
    SecurityAuditConfig get_config() const;
    
    // Check if a specific event type should be logged
    bool should_log_event(SecurityEventType event_type) const;
    
    // Rotate the log file if it exceeds the maximum size
    Result<bool> rotate_log_file();
    
    // Get statistics about the audit log
    Result<std::unordered_map<std::string, std::string>> get_audit_stats() const;
    
    // Search audit logs by various criteria
    Result<std::vector<SecurityEvent>> search_audit_log(
        SecurityEventType event_type = SecurityEventType::AUTHENTICATION_ATTEMPT, 
        const std::string& user_id = "", 
        std::chrono::system_clock::time_point start_time = std::chrono::system_clock::time_point{},
        std::chrono::system_clock::time_point end_time = std::chrono::system_clock::time_point{},
        int limit = 100) const;

private:
    // Internal logging function
    Result<bool> write_log_entry(const SecurityEvent& event);
    
    // Format event as a log entry based on configured format
    std::string format_log_entry(const SecurityEvent& event) const;
    
    // Helper function to escape strings for JSON format
    std::string escape_json_string(const std::string& str) const;
    
    // Check if log rotation is needed
    bool needs_rotation() const;
    
    // Validate configuration
    bool validate_config(const SecurityAuditConfig& config) const;
    
    // Rotate log file with appropriate naming
    Result<bool> perform_log_rotation();
};

} // namespace jadevectordb

#endif // JADEVECTORDB_SECURITY_AUDIT_LOGGER_H