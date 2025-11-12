#ifndef JADEVECTORDB_SECURITY_AUDIT_H
#define JADEVECTORDB_SECURITY_AUDIT_H

#include <string>
#include <memory>
#include <mutex>
#include <fstream>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <deque>

#include "logging.h"
#include "auth.h"

namespace jadevectordb {

enum class AuditEventType {
    USER_LOGIN,
    USER_LOGOUT,
    API_KEY_CREATED,
    API_KEY_REVOKED,
    DATABASE_CREATED,
    DATABASE_DELETED,
    DATABASE_UPDATED,
    VECTORS_ADDED,
    VECTORS_DELETED,
    VECTORS_SEARCHED,
    INDEX_CREATED,
    INDEX_DELETED,
    CONFIG_UPDATED,
    AUTHENTICATION_FAILED,
    UNAUTHORIZED_ACCESS_ATTEMPT,
    PERMISSION_DENIED,
    DATA_EXPORT,
    DATA_IMPORT,
    USER_CREATED,
    USER_DELETED,
    ROLE_CREATED,
    ROLE_DELETED,
    ROLE_ASSIGNED,
    ROLE_UNASSIGNED
};

struct AuditEvent {
    std::string event_id;
    AuditEventType event_type;
    std::string user_id;
    std::string api_key_id;  // The API key used for the operation
    std::string ip_address;
    std::string user_agent;
    std::string resource_id;  // ID of the resource being accessed (database, vector, etc.)
    std::string resource_type;  // Type of the resource (database, vector, index, etc.)
    std::string action;  // The specific action taken
    std::string details;  // Additional details about the action
    std::chrono::system_clock::time_point timestamp;
    bool success;  // Whether the operation was successful
    
    AuditEvent() : success(true) {}
};

class SecurityAuditLogger {
private:
    std::shared_ptr<logging::Logger> logger_;
    mutable std::mutex audit_mutex_;
    std::string audit_log_file_;
    std::ofstream audit_file_stream_;
    size_t max_file_size_;
    size_t current_file_size_;
    AuthManager* auth_manager_;
    
    // Configuration for audit logging
    bool audit_enabled_;
    bool log_user_operations_;
    bool log_authentication_events_;
    bool log_configuration_changes_;
    bool log_data_access_;
    
    // Map to track session information
    std::unordered_map<std::string, std::string> session_to_user_map_; // session_id -> user_id
    mutable std::deque<AuditEvent> recent_events_;
    size_t max_recent_events_ = 500;

public:
    explicit SecurityAuditLogger(const std::string& log_file = "./logs/security_audit.log");
    ~SecurityAuditLogger();
    
    // Initialize the logger
    bool initialize();
    
    // Configuration methods
    void set_audit_enabled(bool enabled) { audit_enabled_ = enabled; }
    void set_log_user_operations(bool enabled) { log_user_operations_ = enabled; }
    void set_log_authentication_events(bool enabled) { log_authentication_events_ = enabled; }
    void set_log_configuration_changes(bool enabled) { log_configuration_changes_ = enabled; }
    void set_log_data_access(bool enabled) { log_data_access_ = enabled; }
    
    // Core audit logging methods
    bool log_audit_event(const AuditEvent& event);
    bool log_user_operation(const std::string& user_id, 
                           const std::string& api_key_id, 
                           const std::string& resource_id, 
                           const std::string& action, 
                           const std::string& details = "");
    bool log_authentication_event(AuditEventType event_type, 
                                 const std::string& user_identifier, 
                                 const std::string& ip_address, 
                                 bool success, 
                                 const std::string& details = "");
    bool log_configuration_change(const std::string& user_id,
                                 const std::string& config_name,
                                 const std::string& old_value,
                                 const std::string& new_value);
    bool log_data_access(const std::string& user_id,
                        const std::string& resource_id,
                        const std::string& access_type,
                        bool success,
                        const std::string& details = "");
    
    // Session tracking
    void track_session(const std::string& session_id, const std::string& user_id);
    void remove_session(const std::string& session_id);
    std::string get_user_for_session(const std::string& session_id) const;
    std::vector<AuditEvent> get_recent_events(size_t limit = 100) const;
    
    // Event creation helpers
    AuditEvent create_audit_event(AuditEventType event_type,
                                 const std::string& user_id,
                                 const std::string& api_key_id,
                                 const std::string& resource_id,
                                 const std::string& action,
                                 const std::string& details = "",
                                 bool success = true);
    
    // File rotation
    void rotate_log_file();
    bool is_log_file_size_exceeded() const;
    
    // Get statistics
    size_t get_total_events_logged() const;
    
private:
    std::string generate_event_id() const;
    std::string format_event_timestamp(const std::chrono::system_clock::time_point& time) const;
    std::string audit_event_type_to_string(AuditEventType type) const;
    void write_event_to_file(const AuditEvent& event);
};

} // namespace jadevectordb

#endif // JADEVECTORDB_SECURITY_AUDIT_H