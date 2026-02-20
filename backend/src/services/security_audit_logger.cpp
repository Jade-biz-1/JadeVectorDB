#include "security_audit_logger.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>

namespace jadevectordb {

SecurityAuditLogger::SecurityAuditLogger() : current_log_size_(0) {
    logger_ = logging::LoggerManager::get_logger("SecurityAuditLogger");
}

SecurityAuditLogger::~SecurityAuditLogger() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

bool SecurityAuditLogger::initialize(const SecurityAuditConfig& config) {
    try {
        if (!validate_config(config)) {
            LOG_ERROR(logger_, "Invalid security audit configuration provided");
            return false;
        }
        
        config_ = config;
        
        // Open the log file
        log_file_.open(config_.log_file_path, std::ios::app | std::ios::out);
        if (!log_file_.is_open()) {
            LOG_ERROR(logger_, "Failed to open security audit log file: " + config_.log_file_path);
            return false;
        }
        
        // Calculate initial log size
        log_file_.seekp(0, std::ios::end);
        current_log_size_ = static_cast<size_t>(log_file_.tellp());
        
        LOG_INFO(logger_, "SecurityAuditLogger initialized with file: " + config_.log_file_path);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in SecurityAuditLogger::initialize: " + std::string(e.what()));
        return false;
    }
}

Result<bool> SecurityAuditLogger::log_security_event(const SecurityEvent& event) {
    try {
        // Check if this event type should be logged
        if (!config_.log_all_operations && !should_log_event(event.event_type)) {
            return true; // Not an error, just not logging this event type
        }
        
        std::lock_guard<std::mutex> lock(log_mutex_);

        // Store in recent events ring buffer
        if (recent_events_.size() >= MAX_RECENT_EVENTS) {
            recent_events_.erase(recent_events_.begin());
        }
        recent_events_.push_back(event);

        auto result = write_log_entry(event);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to write security event to log: " + ErrorHandler::format_error(result.error()));
            return result;
        }
        
        // Check if rotation is needed after writing
        if (needs_rotation()) {
            auto rotate_result = rotate_log_file();
            if (!rotate_result.has_value()) {
                LOG_WARN(logger_, "Log rotation failed: " + ErrorHandler::format_error(rotate_result.error()));
            }
        }
        
        LOG_DEBUG(logger_, "Logged security event: " + std::to_string(static_cast<int>(event.event_type)) + 
                 " for user " + event.user_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in log_security_event: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to log security event: " + std::string(e.what()));
    }
}

Result<bool> SecurityAuditLogger::log_authentication_attempt(const std::string& user_id, 
                                                           const std::string& ip_address,
                                                           bool success,
                                                           const std::string& details) {
    try {
        SecurityEvent event(
            success ? SecurityEventType::AUTHENTICATION_SUCCESS : SecurityEventType::AUTHENTICATION_FAILURE,
            user_id, 
            ip_address, 
            "authentication", 
            "authenticate",
            success
        );
        
        event.details = details;
        return log_security_event(event);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in log_authentication_attempt: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to log authentication attempt: " + std::string(e.what()));
    }
}

Result<bool> SecurityAuditLogger::log_authorization_check(const std::string& user_id,
                                                        const std::string& resource,
                                                        const std::string& operation,
                                                        bool granted,
                                                        const std::string& details) {
    try {
        SecurityEvent event(
            granted ? SecurityEventType::AUTHORIZATION_GRANTED : SecurityEventType::AUTHORIZATION_DENIED,
            user_id, 
            "", // IP address would need to be passed separately if needed
            resource, 
            operation,
            granted
        );
        
        event.details = details;
        return log_security_event(event);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in log_authorization_check: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to log authorization check: " + std::string(e.what()));
    }
}

Result<bool> SecurityAuditLogger::log_data_access(const std::string& user_id,
                                                const std::string& ip_address,
                                                const std::string& database_id,
                                                const std::string& vector_id,
                                                bool success,
                                                const std::string& details) {
    try {
        std::string resource = database_id + ":" + vector_id;
        SecurityEvent event(
            SecurityEventType::DATA_ACCESS,
            user_id, 
            ip_address, 
            resource, 
            "read",
            success
        );
        
        event.details = details;
        return log_security_event(event);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in log_data_access: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to log data access: " + std::string(e.what()));
    }
}

Result<bool> SecurityAuditLogger::log_data_modification(const std::string& user_id,
                                                      const std::string& ip_address,
                                                      const std::string& database_id,
                                                      const std::string& vector_id,
                                                      bool success,
                                                      const std::string& details) {
    try {
        std::string resource = database_id + ":" + vector_id;
        SecurityEvent event(
            SecurityEventType::DATA_MODIFICATION,
            user_id, 
            ip_address, 
            resource, 
            "update",
            success
        );
        
        event.details = details;
        return log_security_event(event);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in log_data_modification: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to log data modification: " + std::string(e.what()));
    }
}

Result<bool> SecurityAuditLogger::log_data_deletion(const std::string& user_id,
                                                   const std::string& ip_address,
                                                   const std::string& database_id,
                                                   const std::string& vector_id,
                                                   bool success,
                                                   const std::string& details) {
    try {
        std::string resource = database_id + ":" + vector_id;
        SecurityEvent event(
            SecurityEventType::DATA_DELETION,
            user_id, 
            ip_address, 
            resource, 
            "delete",
            success
        );
        
        event.details = details;
        return log_security_event(event);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in log_data_deletion: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to log data deletion: " + std::string(e.what()));
    }
}

Result<bool> SecurityAuditLogger::log_configuration_change(const std::string& user_id,
                                                         const std::string& ip_address,
                                                         const std::string& config_change,
                                                         bool success,
                                                         const std::string& details) {
    try {
        SecurityEvent event(
            SecurityEventType::CONFIGURATION_CHANGE,
            user_id, 
            ip_address, 
            "configuration", 
            config_change,
            success
        );
        
        event.details = details;
        return log_security_event(event);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in log_configuration_change: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to log configuration change: " + std::string(e.what()));
    }
}

Result<bool> SecurityAuditLogger::log_security_events(const std::vector<SecurityEvent>& events) {
    try {
        bool all_success = true;
        
        for (const auto& event : events) {
            auto result = log_security_event(event);
            if (!result.has_value()) {
                LOG_WARN(logger_, "Failed to log security event: " + ErrorHandler::format_error(result.error()));
                all_success = false;
            }
        }
        
        return all_success;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in log_security_events: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to log security events: " + std::string(e.what()));
    }
}

Result<bool> SecurityAuditLogger::update_config(const SecurityAuditConfig& new_config) {
    try {
        if (!validate_config(new_config)) {
            LOG_ERROR(logger_, "Invalid security audit configuration provided for update");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid security audit configuration");
        }
        
        std::lock_guard<std::mutex> lock(log_mutex_);
        config_ = new_config;
        
        LOG_INFO(logger_, "Updated security audit configuration");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_config: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to update configuration: " + std::string(e.what()));
    }
}

SecurityAuditConfig SecurityAuditLogger::get_config() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(log_mutex_));
    return config_;
}

bool SecurityAuditLogger::should_log_event(SecurityEventType event_type) const {
    for (const auto& logged_type : config_.events_to_log) {
        if (logged_type == event_type) {
            return true;
        }
    }
    return false;
}

Result<bool> SecurityAuditLogger::rotate_log_file() {
    try {
        std::lock_guard<std::mutex> lock(log_mutex_);
        
        LOG_INFO(logger_, "Rotating security audit log file");
        
        // Close current log file
        if (log_file_.is_open()) {
            log_file_.close();
        }
        
        // Perform the rotation
        auto result = perform_log_rotation();
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to perform log rotation: " + ErrorHandler::format_error(result.error()));
            return result;
        }
        
        // Reopen the log file for continued writing
        log_file_.open(config_.log_file_path, std::ios::app | std::ios::out);
        if (!log_file_.is_open()) {
            LOG_ERROR(logger_, "Failed to reopen security audit log file after rotation: " + config_.log_file_path);
            RETURN_ERROR(ErrorCode::STORAGE_IO_ERROR, "Failed to reopen log file");
        }
        
        current_log_size_ = 0; // Reset after rotation
        
        LOG_INFO(logger_, "Security audit log file rotated successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in rotate_log_file: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to rotate log file: " + std::string(e.what()));
    }
}

Result<std::unordered_map<std::string, std::string>> SecurityAuditLogger::get_audit_stats() const {
    try {
        std::lock_guard<std::mutex> lock(log_mutex_);
        
        std::unordered_map<std::string, std::string> stats;
        stats["log_file_path"] = config_.log_file_path;
        stats["current_size_bytes"] = std::to_string(current_log_size_);
        stats["current_size_mb"] = std::to_string(current_log_size_ / (1024 * 1024));
        stats["max_file_size_mb"] = std::to_string(config_.max_log_file_size_mb);
        stats["enabled"] = config_.enabled ? "true" : "false";
        stats["events_to_log_count"] = std::to_string(config_.events_to_log.size());
        stats["log_format"] = config_.log_format;
        
        LOG_DEBUG(logger_, "Generated security audit statistics");
        return stats;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_audit_stats: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to get audit stats: " + std::string(e.what()));
    }
}

Result<std::vector<SecurityEvent>> SecurityAuditLogger::search_audit_log(
    SecurityEventType event_type, 
    const std::string& user_id, 
    std::chrono::system_clock::time_point start_time,
    std::chrono::system_clock::time_point end_time,
    int limit) const {
    try {
        LOG_DEBUG(logger_, "Searching security audit log with filters");

        std::lock_guard<std::mutex> lock(log_mutex_);
        std::vector<SecurityEvent> results;

        // Search recent events in reverse order (newest first)
        for (auto it = recent_events_.rbegin(); it != recent_events_.rend(); ++it) {
            if (static_cast<int>(results.size()) >= limit) break;

            const auto& event = *it;

            // Filter by user_id if specified
            if (!user_id.empty() && event.user_id != user_id) continue;

            // Filter by time range if specified
            if (start_time.time_since_epoch().count() > 0 && event.timestamp < start_time) continue;
            if (end_time.time_since_epoch().count() > 0 && event.timestamp > end_time) continue;

            results.push_back(event);
        }

        LOG_DEBUG(logger_, "Found " + std::to_string(results.size()) + " events in audit log search");
        return results;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in search_audit_log: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to search audit log: " + std::string(e.what()));
    }
}

// Private methods

Result<bool> SecurityAuditLogger::write_log_entry(const SecurityEvent& event) {
    try {
        std::string log_entry = format_log_entry(event);
        log_file_ << log_entry << std::endl;
        
        if (log_file_.fail()) {
            RETURN_ERROR(ErrorCode::STORAGE_IO_ERROR, "Failed to write to log file");
        }
        
        current_log_size_ += log_entry.length() + 1; // +1 for newline
        
        // Flush to ensure data is written immediately for security purposes
        log_file_.flush();
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in write_log_entry: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::STORAGE_IO_ERROR, "Failed to write log entry: " + std::string(e.what()));
    }
}

std::string SecurityAuditLogger::format_log_entry(const SecurityEvent& event) const {
    if (config_.log_format == "json") {
        // Format as JSON
        std::ostringstream json_stream;
        json_stream << "{"
                   << "\"event_id\":\"" << event.event_id << "\","
                   << "\"event_type\":" << static_cast<int>(event.event_type) << ","
                   << "\"user_id\":\"" << escape_json_string(event.user_id) << "\","
                   << "\"ip_address\":\"" << escape_json_string(event.ip_address) << "\","
                   << "\"resource\":\"" << escape_json_string(event.resource_accessed) << "\","
                   << "\"operation\":\"" << escape_json_string(event.operation) << "\","
                   << "\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(
                       event.timestamp.time_since_epoch()).count() << ","
                   << "\"details\":\"" << escape_json_string(event.details) << "\","
                   << "\"success\":" << (event.success ? "true" : "false") << ","
                   << "\"session_id\":\"" << escape_json_string(event.session_id) << "\","
                   << "\"client_info\":\"" << escape_json_string(event.client_info) << "\""
                   << "}";
        return json_stream.str();
    } else {
        // Default to a simple format
        std::ostringstream log_stream;
        auto time_t = std::chrono::system_clock::to_time_t(event.timestamp);
        log_stream << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
                  << " | " << static_cast<int>(event.event_type)
                  << " | " << event.user_id
                  << " | " << event.ip_address
                  << " | " << event.resource_accessed
                  << " | " << event.operation
                  << " | " << (event.success ? "SUCCESS" : "FAILURE")
                  << " | " << event.details;
        return log_stream.str();
    }
}

std::string SecurityAuditLogger::escape_json_string(const std::string& str) const {
    std::string result;
    result.reserve(str.length()); // Reserve space to minimize allocations
    
    for (char c : str) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (c < 0x20) {
                    result += "\\u";
                    result += "00";
                    result += "0123456789ABCDEF"[c >> 4];
                    result += "0123456789ABCDEF"[c & 0x0F];
                } else {
                    result += c;
                }
                break;
        }
    }
    
    return result;
}

bool SecurityAuditLogger::needs_rotation() const {
    size_t max_size_bytes = static_cast<size_t>(config_.max_log_file_size_mb) * 1024 * 1024;
    return current_log_size_ >= max_size_bytes;
}

Result<bool> SecurityAuditLogger::perform_log_rotation() {
    try {
        // In a real implementation, this would move the current log file to an archive
        // with a timestamp or sequence number, and potentially delete older archives
        // to keep only the configured number of log files.
        
        // For this implementation, we'll just note that rotation is needed
        LOG_DEBUG(logger_, "Performed log rotation (simulated)");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in perform_log_rotation: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::INTERNAL_ERROR, "Failed to perform log rotation: " + std::string(e.what()));
    }
}

bool SecurityAuditLogger::validate_config(const SecurityAuditConfig& config) const {
    // Basic validation
    if (config.max_log_file_size_mb <= 0) {
        LOG_ERROR(logger_, "Invalid max log file size: " + std::to_string(config.max_log_file_size_mb));
        return false;
    }
    
    if (config.max_log_files <= 0) {
        LOG_ERROR(logger_, "Invalid max log files: " + std::to_string(config.max_log_files));
        return false;
    }
    
    if (config.log_file_path.empty()) {
        LOG_ERROR(logger_, "Log file path cannot be empty");
        return false;
    }
    
    if (config.log_format != "json" && config.log_format != "simple") {
        LOG_ERROR(logger_, "Invalid log format: " + config.log_format);
        return false;
    }
    
    return true;
}

} // namespace jadevectordb