#include "security_audit.h"
#include <iomanip>
#include <algorithm>
#include <random>

namespace jadevectordb {

SecurityAuditLogger::SecurityAuditLogger(const std::string& log_file)
    : audit_log_file_(log_file), 
      max_file_size_(50 * 1024 * 1024), // 50MB default
      current_file_size_(0),
      audit_enabled_(true),
      log_user_operations_(true),
      log_authentication_events_(true),
      log_configuration_changes_(true),
      log_data_access_(true) {

    logger_ = logging::LoggerManager::get_logger("SecurityAuditLogger");
    // REMOVED: auth_manager_ = AuthManager::get_instance() - migrated to AuthenticationService
}

SecurityAuditLogger::~SecurityAuditLogger() {
    if (audit_file_stream_.is_open()) {
        audit_file_stream_.close();
    }
}

bool SecurityAuditLogger::initialize() {
    try {
        audit_file_stream_.open(audit_log_file_, std::ios::app);
        if (!audit_file_stream_.is_open()) {
            LOG_ERROR(logger_, "Failed to open audit log file: " + audit_log_file_);
            return false;
        }
        
        // Get current file size
        audit_file_stream_.seekp(0, std::ios::end);
        current_file_size_ = audit_file_stream_.tellp();
        
        LOG_INFO(logger_, "Security audit logger initialized. Log file: " + audit_log_file_);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Error initializing security audit logger: " + std::string(e.what()));
        return false;
    }
}

bool SecurityAuditLogger::log_audit_event(const AuditEvent& event) {
    if (!audit_enabled_) {
        return true; // If auditing is disabled, just return success
    }
    
    std::lock_guard<std::mutex> lock(audit_mutex_);
    
    try {
        // Write to file
        write_event_to_file(event);
        // Track recent events in memory for quick retrieval
        recent_events_.push_back(event);
        if (recent_events_.size() > max_recent_events_) {
            recent_events_.pop_front();
        }
        
        // Log to application logger as well
        std::string event_type_str = audit_event_type_to_string(event.event_type);
        std::string status = event.success ? "SUCCESS" : "FAILED";
        
        LOG_INFO(logger_, "AUDIT [" + status + "] " + event_type_str + 
                          " - User: " + event.user_id + 
                          " - Resource: " + event.resource_id + 
                          " - Action: " + event.action);
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Error logging audit event: " + std::string(e.what()));
        return false;
    }
}

bool SecurityAuditLogger::log_user_operation(const std::string& user_id, 
                                            const std::string& api_key_id,
                                            const std::string& resource_id, 
                                            const std::string& action, 
                                            const std::string& details) {
    if (!log_user_operations_ || !audit_enabled_) {
        return true;
    }
    
    AuditEvent event = create_audit_event(AuditEventType::USER_LOGIN, 
                                         user_id, api_key_id, 
                                         resource_id, action, details);
    return log_audit_event(event);
}

bool SecurityAuditLogger::log_authentication_event(AuditEventType event_type, 
                                                  const std::string& user_identifier, 
                                                  const std::string& ip_address, 
                                                  bool success, 
                                                  const std::string& details) {
    if (!log_authentication_events_ || !audit_enabled_) {
        return true;
    }
    
    AuditEvent event;
    event.event_type = event_type;
    event.event_id = generate_event_id();
    event.user_id = user_identifier; // This might be a user ID or username
    event.ip_address = ip_address;
    event.timestamp = std::chrono::system_clock::now();
    event.success = success;
    event.details = details;
    
    return log_audit_event(event);
}

bool SecurityAuditLogger::log_configuration_change(const std::string& user_id,
                                                  const std::string& config_name,
                                                  const std::string& old_value,
                                                  const std::string& new_value) {
    if (!log_configuration_changes_ || !audit_enabled_) {
        return true;
    }
    
    std::string details = "Config: " + config_name + 
                         ", Old Value: " + old_value + 
                         ", New Value: " + new_value;
    
    AuditEvent event = create_audit_event(AuditEventType::CONFIG_UPDATED,
                                         user_id, "", "config:" + config_name,
                                         "UPDATE", details);
    return log_audit_event(event);
}

bool SecurityAuditLogger::log_data_access(const std::string& user_id,
                                         const std::string& resource_id,
                                         const std::string& access_type,
                                         bool success,
                                         const std::string& details) {
    if (!log_data_access_ || !audit_enabled_) {
        return true;
    }
    
    AuditEvent event = create_audit_event(AuditEventType::VECTORS_SEARCHED,
                                         user_id, "", resource_id,
                                         access_type, details, success);
    return log_audit_event(event);
}

void SecurityAuditLogger::track_session(const std::string& session_id, const std::string& user_id) {
    std::lock_guard<std::mutex> lock(audit_mutex_);
    session_to_user_map_[session_id] = user_id;
}

void SecurityAuditLogger::remove_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(audit_mutex_);
    session_to_user_map_.erase(session_id);
}

std::string SecurityAuditLogger::get_user_for_session(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(audit_mutex_);
    auto it = session_to_user_map_.find(session_id);
    if (it != session_to_user_map_.end()) {
        return it->second;
    }
    return "";
}

std::vector<AuditEvent> SecurityAuditLogger::get_recent_events(size_t limit) const {
    std::lock_guard<std::mutex> lock(audit_mutex_);
    std::vector<AuditEvent> events;
    const size_t count = std::min(limit, recent_events_.size());
    events.reserve(count);
    auto start = recent_events_.end();
    std::advance(start, -static_cast<long>(count));
    events.insert(events.end(), start, recent_events_.end());
    return events;
}

AuditEvent SecurityAuditLogger::create_audit_event(AuditEventType event_type,
                                                   const std::string& user_id,
                                                   const std::string& api_key_id,
                                                   const std::string& resource_id,
                                                   const std::string& action,
                                                   const std::string& details,
                                                   bool success) {
    AuditEvent event;
    event.event_type = event_type;
    event.event_id = generate_event_id();
    event.user_id = user_id;
    event.api_key_id = api_key_id;
    event.resource_id = resource_id;
    event.action = action;
    event.details = details;
    event.timestamp = std::chrono::system_clock::now();
    event.success = success;
    
    // Set resource type based on resource_id
    if (resource_id.find("db_") == 0) {
        event.resource_type = "database";
    } else if (resource_id.find("vec_") == 0) {
        event.resource_type = "vector";
    } else if (resource_id.find("idx_") == 0) {
        event.resource_type = "index";
    } else if (resource_id.find("user_") == 0) {
        event.resource_type = "user";
    } else if (resource_id.find("role_") == 0) {
        event.resource_type = "role";
    } else {
        event.resource_type = "general";
    }
    
    return event;
}

void SecurityAuditLogger::rotate_log_file() {
    std::lock_guard<std::mutex> lock(audit_mutex_);
    
    if (audit_file_stream_.is_open()) {
        audit_file_stream_.close();
    }
    
    // Create a backup file with timestamp
    std::string timestamp = format_event_timestamp(std::chrono::system_clock::now());
    // Replace colons with hyphens for filename safety
    std::string safe_timestamp = timestamp;
    std::replace(safe_timestamp.begin(), safe_timestamp.end(), ':', '-');
    
    std::string backup_file = audit_log_file_ + "." + safe_timestamp + ".bak";
    
    // Rename current log file to backup
    std::rename(audit_log_file_.c_str(), backup_file.c_str());
    
    // Open new log file
    audit_file_stream_.open(audit_log_file_, std::ios::out);
    current_file_size_ = 0;
    
    LOG_INFO(logger_, "Log file rotated. Backup saved as: " + backup_file);
}

bool SecurityAuditLogger::is_log_file_size_exceeded() const {
    return current_file_size_ >= max_file_size_;
}

size_t SecurityAuditLogger::get_total_events_logged() const {
    // In a real implementation, this would track the count
    // For now, we'll return 0 since we don't maintain a persistent count
    return 0;
}

std::string SecurityAuditLogger::generate_event_id() const {
    // Generate a unique event ID
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto count = duration.count();
    
    std::stringstream ss;
    ss << std::hex << count;
    
    // Add some randomness to ensure uniqueness
    thread_local static std::random_device rd;
    thread_local static std::mt19937 gen(rd());
    thread_local static std::uniform_int_distribution<> dis(1000, 9999);
    
    ss << "_" << dis(gen);
    return ss.str();
}

std::string SecurityAuditLogger::format_event_timestamp(const std::chrono::system_clock::time_point& time) const {
    auto time_t = std::chrono::system_clock::to_time_t(time);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z";
    return ss.str();
}

std::string SecurityAuditLogger::audit_event_type_to_string(AuditEventType type) const {
    switch (type) {
        case AuditEventType::USER_LOGIN: return "USER_LOGIN";
        case AuditEventType::USER_LOGOUT: return "USER_LOGOUT";
        case AuditEventType::API_KEY_CREATED: return "API_KEY_CREATED";
        case AuditEventType::API_KEY_REVOKED: return "API_KEY_REVOKED";
        case AuditEventType::DATABASE_CREATED: return "DATABASE_CREATED";
        case AuditEventType::DATABASE_DELETED: return "DATABASE_DELETED";
        case AuditEventType::DATABASE_UPDATED: return "DATABASE_UPDATED";
        case AuditEventType::VECTORS_ADDED: return "VECTORS_ADDED";
        case AuditEventType::VECTORS_DELETED: return "VECTORS_DELETED";
        case AuditEventType::VECTORS_SEARCHED: return "VECTORS_SEARCHED";
        case AuditEventType::INDEX_CREATED: return "INDEX_CREATED";
        case AuditEventType::INDEX_DELETED: return "INDEX_DELETED";
        case AuditEventType::CONFIG_UPDATED: return "CONFIG_UPDATED";
        case AuditEventType::AUTHENTICATION_FAILED: return "AUTHENTICATION_FAILED";
        case AuditEventType::UNAUTHORIZED_ACCESS_ATTEMPT: return "UNAUTHORIZED_ACCESS_ATTEMPT";
        case AuditEventType::PERMISSION_DENIED: return "PERMISSION_DENIED";
        case AuditEventType::DATA_EXPORT: return "DATA_EXPORT";
        case AuditEventType::DATA_IMPORT: return "DATA_IMPORT";
        case AuditEventType::USER_CREATED: return "USER_CREATED";
        case AuditEventType::USER_DELETED: return "USER_DELETED";
        case AuditEventType::ROLE_CREATED: return "ROLE_CREATED";
        case AuditEventType::ROLE_DELETED: return "ROLE_DELETED";
        case AuditEventType::ROLE_ASSIGNED: return "ROLE_ASSIGNED";
        case AuditEventType::ROLE_UNASSIGNED: return "ROLE_UNASSIGNED";
        default: return "UNKNOWN_EVENT";
    }
}

void SecurityAuditLogger::write_event_to_file(const AuditEvent& event) {
    std::stringstream ss;
    
    // Log in JSON format for better parsing
    ss << "{";
    ss << "\"eventId\":\"" << event.event_id << "\",";
    ss << "\"eventType\":\"" << audit_event_type_to_string(event.event_type) << "\",";
    ss << "\"userId\":\"" << event.user_id << "\",";
    ss << "\"apiKeyId\":\"" << event.api_key_id << "\",";
    ss << "\"ipAddress\":\"" << event.ip_address << "\",";
    ss << "\"resourceId\":\"" << event.resource_id << "\",";
    ss << "\"resourceType\":\"" << event.resource_type << "\",";
    ss << "\"action\":\"" << event.action << "\",";
    ss << "\"details\":\"" << event.details << "\",";
    ss << "\"timestamp\":\"" << format_event_timestamp(event.timestamp) << "\",";
    ss << "\"success\":" << (event.success ? "true" : "false");
    ss << "}" << std::endl;
    
    std::string log_line = ss.str();
    
    if (is_log_file_size_exceeded()) {
        rotate_log_file();
    }
    
    if (audit_file_stream_.is_open()) {
        audit_file_stream_ << log_line;
        audit_file_stream_.flush();
        current_file_size_ += log_line.length();
    } else {
        // If we can't write to file, at least log to the application logger
        LOG_WARN(logger_, "Could not write audit event to file: " + log_line);
    }
}

} // namespace jadevectordb