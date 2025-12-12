#ifndef JADEVECTORDB_ALERT_SERVICE_H
#define JADEVECTORDB_ALERT_SERVICE_H

#include "services/metrics_service.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <functional>

namespace jadevectordb {

// Represents a configured alert rule
struct AlertRule {
    std::string rule_id;
    std::string name;
    std::string description;
    std::string metric_name;
    std::string condition;      // "gt", "lt", "eq", "gte", "lte"
    double threshold;
    std::string duration;       // Duration before alert triggers (e.g., "5m", "10s", "1h")
    std::vector<std::string> channels;  // Where to send alerts
    std::unordered_map<std::string, std::string> labels;  // Filter metrics by labels
    bool enabled;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    
    AlertRule() : threshold(0.0), enabled(true) {}
    AlertRule(const std::string& id, const std::string& n, const std::string& desc)
        : rule_id(id), name(n), description(desc), enabled(true),
          created_at(std::chrono::system_clock::now()),
          updated_at(std::chrono::system_clock::now()) {}
};

// Represents an active alert
struct ActiveAlert {
    std::string alert_id;
    std::string rule_id;
    std::string name;
    std::string status;  // "firing", "pending", "resolved"
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    std::chrono::system_clock::time_point updated_at;
    double value;        // Value that triggered the alert
    std::unordered_map<std::string, std::string> labels;
    std::string message;
    bool resolved;
    
    ActiveAlert() : value(0.0), resolved(false) {}
    ActiveAlert(const std::string& id, const std::string& rule_id, const std::string& n)
        : alert_id(id), rule_id(rule_id), name(n), status("firing"),
          start_time(std::chrono::system_clock::now()), resolved(false) {}
};

// Configuration for alerting
struct AlertConfig {
    bool enabled = true;                    // Whether alerting is enabled
    int evaluation_interval_seconds = 30;  // How often to evaluate alert rules
    int max_active_alerts = 1000;          // Maximum number of active alerts to track
    std::string default_severity = "warning"; // Default severity level
    std::vector<std::string> notification_channels;  // Default notification channels
    
    AlertConfig() = default;
};

/**
 * @brief Service to manage alerts based on metrics thresholds
 * 
 * This service monitors metrics and triggers alerts when thresholds are exceeded,
 * supporting various notification channels and alert management capabilities.
 */
class AlertService {
private:
    std::shared_ptr<logging::Logger> logger_;
    std::shared_ptr<MetricsService> metrics_service_;
    AlertConfig config_;
    
    std::vector<AlertRule> alert_rules_;
    std::vector<ActiveAlert> active_alerts_;
    std::unordered_map<std::string, std::chrono::system_clock::time_point> rule_last_evaluated_;

    mutable std::mutex alert_mutex_;
    
public:
    explicit AlertService(std::shared_ptr<MetricsService> metrics_service = nullptr);
    ~AlertService() = default;
    
    // Initialize the alert service with configuration
    bool initialize(const AlertConfig& config);
    
    // Add a new alert rule
    Result<std::string> add_alert_rule(const AlertRule& rule);
    
    // Update an existing alert rule
    Result<bool> update_alert_rule(const std::string& rule_id, const AlertRule& rule);
    
    // Remove an alert rule
    Result<bool> remove_alert_rule(const std::string& rule_id);
    
    // Get all alert rules
    Result<std::vector<AlertRule>> get_alert_rules() const;
    
    // Get alert rule by ID
    Result<AlertRule> get_alert_rule(const std::string& rule_id) const;
    
    // Evaluate all alert rules against current metrics
    Result<bool> evaluate_alert_rules();
    
    // Evaluate a specific alert rule
    Result<bool> evaluate_alert_rule(const std::string& rule_id);
    
    // Get all active alerts
    Result<std::vector<ActiveAlert>> get_active_alerts() const;
    
    // Get active alerts by rule
    Result<std::vector<ActiveAlert>> get_active_alerts_by_rule(const std::string& rule_id) const;
    
    // Get active alerts by status
    Result<std::vector<ActiveAlert>> get_active_alerts_by_status(const std::string& status) const;
    
    // Resolve an active alert
    Result<bool> resolve_alert(const std::string& alert_id);
    
    // Send alert to configured channels
    Result<bool> send_alert(const ActiveAlert& alert);
    
    // Get alert statistics
    Result<std::unordered_map<std::string, int>> get_alert_stats() const;
    
    // Update alert configuration
    Result<bool> update_config(const AlertConfig& new_config);
    
    // Silence alerts matching certain criteria
    Result<bool> silence_alerts(const std::unordered_map<std::string, std::string>& match_labels,
                              const std::chrono::system_clock::time_point& until);
    
    // Get all configured notification channels
    std::vector<std::string> get_notification_channels() const;

private:
    // Internal helper methods
    
    // Check if a metric value meets the alert condition
    bool check_metric_condition(double value, const AlertRule& rule) const;
    
    // Parse duration string (e.g., "5m", "10s", "1h") to seconds
    std::chrono::seconds parse_duration(const std::string& duration_str) const;
    
    // Validate alert rule configuration
    bool validate_alert_rule(const AlertRule& rule) const;
    
    // Create an active alert from a rule and metric value
    ActiveAlert create_active_alert(const AlertRule& rule, double value) const;
    
    // Clean up resolved or expired alerts
    void cleanup_inactive_alerts();
    
    // Send alert to log channel
    Result<bool> send_alert_to_log(const ActiveAlert& alert);
    
    // Send alert to webhook channel
    Result<bool> send_alert_to_webhook(const ActiveAlert& alert);
    
    // Send alert to email channel
    Result<bool> send_alert_to_email(const ActiveAlert& alert);
    
    // Check if an alert should be firing based on duration
    bool should_alert_be_firing(const AlertRule& rule, double current_value) const;
    
    // Generate alert message from rule and value
    std::string generate_alert_message(const AlertRule& rule, double value) const;
    
    // Update the last evaluation time for a rule
    void update_rule_evaluation_time(const std::string& rule_id);
    
    // Check if enough time has passed since last evaluation
    bool should_evaluate_rule(const std::string& rule_id) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_ALERT_SERVICE_H