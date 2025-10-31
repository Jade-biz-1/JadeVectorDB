#include "alert_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <random>
#include <chrono>

namespace jadevectordb {

AlertService::AlertService() {
    logger_ = logging::LoggerManager::get_logger("AlertService");
}

bool AlertService::initialize(const AlertConfig& config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (!validate_config(config)) {
            LOG_ERROR(logger_, "Invalid alert configuration provided");
            return false;
        }
        
        config_ = config;
        
        LOG_INFO(logger_, "AlertService initialized with threshold: " + 
                std::to_string(config_.alert_threshold) + 
                ", cooldown_period: " + std::to_string(config_.cooldown_period_seconds) + "s");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in AlertService::initialize: " + std::string(e.what()));
        return false;
    }
}

Result<bool> AlertService::send_alert(const Alert& alert) {
    try {
        LOG_INFO(logger_, "Sending alert: " + alert.alert_type + " - " + alert.message);
        
        // Check if alert is within cooldown period
        {
            std::lock_guard<std::mutex> lock(alert_mutex_);
            auto now = std::chrono::steady_clock::now();
            
            // Check if we've sent an alert of this type recently
            auto it = last_alert_times_.find(alert.alert_type);
            if (it != last_alert_times_.end()) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - it->second).count();
                if (elapsed < config_.cooldown_period_seconds) {
                    LOG_DEBUG(logger_, "Alert " + alert.alert_type + " is within cooldown period, skipping");
                    return true; // Not an error, just skipping due to cooldown
                }
            }
            
            // Update last alert time
            last_alert_times_[alert.alert_type] = now;
        }
        
        // In a real implementation, this would:
        // 1. Format the alert according to configuration
        // 2. Send it through the appropriate channels (email, Slack, etc.)
        // 3. Log the alert sending
        
        // For now, we'll just log that the alert was sent
        LOG_ALERT(logger_, "ALERT [" + alert.alert_type + "]: " + alert.message + 
                 " (Severity: " + std::to_string(static_cast<int>(alert.severity)) + 
                 ", Source: " + alert.source + ")");
        
        // Send to alert handlers
        {
            std::lock_guard<std::mutex> lock(handler_mutex_);
            for (const auto& handler : alert_handlers_) {
                try {
                    handler(alert);
                } catch (const std::exception& e) {
                    LOG_ERROR(logger_, "Exception in alert handler: " + std::string(e.what()));
                }
            }
        }
        
        LOG_DEBUG(logger_, "Alert sent successfully: " + alert.alert_type);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in send_alert: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to send alert: " + std::string(e.what()));
    }
}

Result<bool> AlertService::register_alert_handler(const AlertHandler& handler) {
    try {
        std::lock_guard<std::mutex> lock(handler_mutex_);
        alert_handlers_.push_back(handler);
        
        LOG_DEBUG(logger_, "Registered new alert handler, total handlers: " + 
                 std::to_string(alert_handlers_.size()));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in register_alert_handler: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to register alert handler: " + std::string(e.what()));
    }
}

Result<bool> AlertService::unregister_alert_handler(const AlertHandler& handler) {
    try {
        std::lock_guard<std::mutex> lock(handler_mutex_);
        
        auto it = std::find(alert_handlers_.begin(), alert_handlers_.end(), handler);
        if (it != alert_handlers_.end()) {
            alert_handlers_.erase(it);
            LOG_DEBUG(logger_, "Unregistered alert handler, total handlers: " + 
                     std::to_string(alert_handlers_.size()));
            return true;
        }
        
        LOG_WARN(logger_, "Alert handler not found for unregistration");
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in unregister_alert_handler: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to unregister alert handler: " + std::string(e.what()));
    }
}

Result<AlertStats> AlertService::get_alert_statistics() const {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);
        
        AlertStats stats;
        stats.total_alerts_sent = alert_counter_;
        stats.active_alerts = static_cast<int>(last_alert_times_.size());
        
        // Calculate alerts per minute
        if (!last_alert_times_.empty()) {
            auto now = std::chrono::steady_clock::now();
            int recent_alerts = 0;
            
            for (const auto& entry : last_alert_times_) {
                auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - entry.second).count();
                if (elapsed <= 1) { // Alerts in the last minute
                    recent_alerts++;
                }
            }
            
            stats.alerts_per_minute = recent_alerts;
        }
        
        LOG_DEBUG(logger_, "Generated alert statistics");
        return stats;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_alert_statistics: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get alert statistics: " + std::string(e.what()));
    }
}

Result<bool> AlertService::clear_alert_history() {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);
        last_alert_times_.clear();
        alert_counter_ = 0;
        
        LOG_INFO(logger_, "Cleared alert history");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in clear_alert_history: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to clear alert history: " + std::string(e.what()));
    }
}

Result<bool> AlertService::update_alert_config(const AlertConfig& new_config) {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (!validate_config(new_config)) {
            LOG_ERROR(logger_, "Invalid alert configuration provided for update");
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid alert configuration");
        }
        
        config_ = new_config;
        
        LOG_INFO(logger_, "Updated alert configuration: threshold=" + 
                std::to_string(config_.alert_threshold) + 
                ", cooldown_period=" + std::to_string(config_.cooldown_period_seconds) + "s");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_alert_config: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update alert configuration: " + std::string(e.what()));
    }
}

AlertConfig AlertService::get_config() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

Result<bool> AlertService::check_threshold_and_alert(const std::string& metric_name, 
                                                  double value, 
                                                  const std::string& database_id) {
    try {
        LOG_DEBUG(logger_, "Checking threshold for metric " + metric_name + ": " + std::to_string(value));
        
        // Check if value exceeds threshold
        if (value > config_.alert_threshold) {
            Alert alert;
            alert.alert_type = "THRESHOLD_EXCEEDED";
            alert.message = "Metric " + metric_name + " exceeded threshold. Value: " + 
                           std::to_string(value) + ", Threshold: " + std::to_string(config_.alert_threshold);
            alert.severity = AlertSeverity::WARNING;
            alert.source = database_id.empty() ? "system" : "database:" + database_id;
            alert.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            auto result = send_alert(alert);
            if (!result.has_value()) {
                LOG_ERROR(logger_, "Failed to send threshold alert: " + 
                         ErrorHandler::format_error(result.error()));
                return result;
            }
            
            LOG_WARN(logger_, "Sent threshold alert for metric " + metric_name + 
                    ": " + alert.message);
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in check_threshold_and_alert: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check threshold and alert: " + std::string(e.what()));
    }
}

Result<bool> AlertService::send_system_alert(const std::string& alert_type, 
                                          const std::string& message, 
                                          AlertSeverity severity) {
    try {
        Alert alert;
        alert.alert_type = alert_type;
        alert.message = message;
        alert.severity = severity;
        alert.source = "system";
        alert.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        auto result = send_alert(alert);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to send system alert: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_INFO(logger_, "Sent system alert [" + alert_type + "]: " + message);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in send_system_alert: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to send system alert: " + std::string(e.what()));
    }
}

Result<bool> AlertService::send_database_alert(const std::string& database_id,
                                            const std::string& alert_type,
                                            const std::string& message,
                                            AlertSeverity severity) {
    try {
        Alert alert;
        alert.alert_type = alert_type;
        alert.message = message;
        alert.severity = severity;
        alert.source = "database:" + database_id;
        alert.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        auto result = send_alert(alert);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to send database alert: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_INFO(logger_, "Sent database alert [" + alert_type + "] for database " + database_id + 
                ": " + message);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in send_database_alert: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to send database alert: " + std::string(e.what()));
    }
}

Result<bool> AlertService::send_cluster_alert(const std::string& node_id,
                                           const std::string& alert_type,
                                           const std::string& message,
                                           AlertSeverity severity) {
    try {
        Alert alert;
        alert.alert_type = alert_type;
        alert.message = message;
        alert.severity = severity;
        alert.source = "cluster:" + node_id;
        alert.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        auto result = send_alert(alert);
        if (!result.has_value()) {
            LOG_ERROR(logger_, "Failed to send cluster alert: " + 
                     ErrorHandler::format_error(result.error()));
            return result;
        }
        
        LOG_INFO(logger_, "Sent cluster alert [" + alert_type + "] for node " + node_id + 
                ": " + message);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in send_cluster_alert: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to send cluster alert: " + std::string(e.what()));
    }
}

Result<bool> AlertService::is_alert_enabled(const std::string& alert_type) const {
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        // Check if alert type is in the disabled list
        if (std::find(config_.disabled_alert_types.begin(), 
                     config_.disabled_alert_types.end(), 
                     alert_type) != config_.disabled_alert_types.end()) {
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in is_alert_enabled: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to check if alert is enabled: " + std::string(e.what()));
    }
}

Result<std::vector<Alert>> AlertService::get_recent_alerts(int limit) const {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);
        
        // In a real implementation, we would maintain a history of alerts
        // For now, we'll return an empty list
        std::vector<Alert> recent_alerts;
        
        LOG_DEBUG(logger_, "Retrieved " + std::to_string(recent_alerts.size()) + 
                 " recent alerts (limit: " + std::to_string(limit) + ")");
        return recent_alerts;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_recent_alerts: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get recent alerts: " + std::string(e.what()));
    }
}

// Private methods

bool AlertService::validate_config(const AlertConfig& config) const {
    // Basic validation
    if (config.alert_threshold < 0) {
        LOG_ERROR(logger_, "Invalid alert threshold: " + std::to_string(config.alert_threshold));
        return false;
    }
    
    if (config.cooldown_period_seconds < 0) {
        LOG_ERROR(logger_, "Invalid cooldown period: " + std::to_string(config.cooldown_period_seconds));
        return false;
    }
    
    if (config.max_alert_history < 0) {
        LOG_ERROR(logger_, "Invalid max alert history: " + std::to_string(config.max_alert_history));
        return false;
    }
    
    return true;
}

void AlertService::increment_alert_counter() {
    std::lock_guard<std::mutex> lock(alert_mutex_);
    alert_counter_++;
}

std::string AlertService::format_alert_message(const Alert& alert) const {
    std::string severity_str;
    switch (alert.severity) {
        case AlertSeverity::INFO: severity_str = "INFO"; break;
        case AlertSeverity::WARNING: severity_str = "WARNING"; break;
        case AlertSeverity::CRITICAL: severity_str = "CRITICAL"; break;
        case AlertSeverity::EMERGENCY: severity_str = "EMERGENCY"; break;
        default: severity_str = "UNKNOWN";
    }
    
    return "[" + severity_str + "] " + alert.alert_type + ": " + alert.message + 
           " (Source: " + alert.source + ")";
}

void AlertService::log_alert(const Alert& alert) const {
    switch (alert.severity) {
        case AlertSeverity::INFO:
            LOG_INFO(logger_, format_alert_message(alert));
            break;
        case AlertSeverity::WARNING:
            LOG_WARN(logger_, format_alert_message(alert));
            break;
        case AlertSeverity::CRITICAL:
            LOG_ERROR(logger_, format_alert_message(alert));
            break;
        case AlertSeverity::EMERGENCY:
            LOG_FATAL(logger_, format_alert_message(alert));
            break;
        default:
            LOG_DEBUG(logger_, format_alert_message(alert));
            break;
    }
}

bool AlertService::should_send_alert(const Alert& alert) const {
    std::lock_guard<std::mutex> lock(alert_mutex_);
    
    // Check if alert type is disabled
    if (!is_alert_enabled(alert.alert_type).value_or(true)) {
        return false;
    }
    
    // Check cooldown period
    auto now = std::chrono::steady_clock::now();
    auto it = last_alert_times_.find(alert.alert_type);
    if (it != last_alert_times_.end()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - it->second).count();
        if (elapsed < config_.cooldown_period_seconds) {
            return false;
        }
    }
    
    return true;
}

void AlertService::update_alert_history(const Alert& alert) {
    std::lock_guard<std::mutex> lock(alert_mutex_);
    
    // Update last alert time
    last_alert_times_[alert.alert_type] = std::chrono::steady_clock::now();
    
    // Increment counter
    alert_counter_++;
    
    // In a real implementation, we would also maintain alert history
    // For now, we'll just keep track of the last alert times
}

} // namespace jadevectordb