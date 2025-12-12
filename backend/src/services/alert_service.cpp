#include "alert_service.h"
#include "lib/logging.h"
#include "lib/error_handling.h"
#include <algorithm>
#include <sstream>
#include <regex>

namespace jadevectordb {

AlertService::AlertService(std::shared_ptr<MetricsService> metrics_service)
    : metrics_service_(metrics_service) {
    logger_ = logging::LoggerManager::get_logger("AlertService");
}

bool AlertService::initialize(const AlertConfig& config) {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);

        if (config.evaluation_interval_seconds <= 0) {
            LOG_ERROR(logger_, "Invalid evaluation interval: " + std::to_string(config.evaluation_interval_seconds));
            return false;
        }

        if (config.max_active_alerts <= 0) {
            LOG_ERROR(logger_, "Invalid max active alerts: " + std::to_string(config.max_active_alerts));
            return false;
        }

        config_ = config;

        LOG_INFO(logger_, "AlertService initialized with evaluation_interval: " +
                std::to_string(config_.evaluation_interval_seconds) + "s, " +
                "max_active_alerts: " + std::to_string(config_.max_active_alerts));
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in AlertService::initialize: " + std::string(e.what()));
        return false;
    }
}

Result<std::string> AlertService::add_alert_rule(const AlertRule& rule) {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);

        if (!validate_alert_rule(rule)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid alert rule configuration");
        }

        std::string rule_id = "rule_" + std::to_string(alert_rules_.size() + 1);
        AlertRule new_rule = rule;
        new_rule.rule_id = rule_id;
        new_rule.created_at = std::chrono::system_clock::now();
        new_rule.updated_at = std::chrono::system_clock::now();

        alert_rules_.push_back(new_rule);
        LOG_INFO(logger_, "Added alert rule: " + rule_id + " (" + rule.name + ")");
        return rule_id;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in add_alert_rule: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to add alert rule: " + std::string(e.what()));
    }
}

Result<bool> AlertService::update_alert_rule(const std::string& rule_id, const AlertRule& rule) {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);

        auto it = std::find_if(alert_rules_.begin(), alert_rules_.end(),
            [&rule_id](const AlertRule& r) { return r.rule_id == rule_id; });

        if (it == alert_rules_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Alert rule not found: " + rule_id);
        }

        if (!validate_alert_rule(rule)) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid alert rule configuration");
        }

        AlertRule updated_rule = rule;
        updated_rule.rule_id = rule_id;
        updated_rule.created_at = it->created_at;
        updated_rule.updated_at = std::chrono::system_clock::now();

        *it = updated_rule;
        LOG_INFO(logger_, "Updated alert rule: " + rule_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_alert_rule: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update alert rule: " + std::string(e.what()));
    }
}

Result<bool> AlertService::remove_alert_rule(const std::string& rule_id) {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);

        auto it = std::find_if(alert_rules_.begin(), alert_rules_.end(),
            [&rule_id](const AlertRule& r) { return r.rule_id == rule_id; });

        if (it == alert_rules_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Alert rule not found: " + rule_id);
        }

        alert_rules_.erase(it);
        LOG_INFO(logger_, "Removed alert rule: " + rule_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in remove_alert_rule: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to remove alert rule: " + std::string(e.what()));
    }
}

Result<std::vector<AlertRule>> AlertService::get_alert_rules() const {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);
        std::vector<AlertRule> rules = alert_rules_;
        LOG_DEBUG(logger_, "Retrieved " + std::to_string(rules.size()) + " alert rules");
        return rules;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_alert_rules: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get alert rules: " + std::string(e.what()));
    }
}

Result<AlertRule> AlertService::get_alert_rule(const std::string& rule_id) const {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);

        auto it = std::find_if(alert_rules_.begin(), alert_rules_.end(),
            [&rule_id](const AlertRule& r) { return r.rule_id == rule_id; });

        if (it == alert_rules_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Alert rule not found: " + rule_id);
        }

        return *it;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_alert_rule: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get alert rule: " + std::string(e.what()));
    }
}

Result<bool> AlertService::evaluate_alert_rules() {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);

        if (!metrics_service_) {
            LOG_WARN(logger_, "Cannot evaluate alert rules: metrics service not available");
            return true;
        }

        for (auto& rule : alert_rules_) {
            if (!rule.enabled) {
                continue;
            }

            // Check if enough time has passed since last evaluation
            if (!should_evaluate_rule(rule.rule_id)) {
                continue;
            }

            // Evaluate the rule
            auto eval_result = evaluate_alert_rule(rule.rule_id);
            if (!eval_result.has_value()) {
                LOG_WARN(logger_, "Failed to evaluate rule " + rule.rule_id + ": " +
                        ErrorHandler::format_error(eval_result.error()));
            }
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in evaluate_alert_rules: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to evaluate alert rules: " + std::string(e.what()));
    }
}

Result<bool> AlertService::evaluate_alert_rule(const std::string& rule_id) {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);

        // Find the rule
        auto rule_it = std::find_if(alert_rules_.begin(), alert_rules_.end(),
            [&rule_id](const AlertRule& r) { return r.rule_id == rule_id; });

        if (rule_it == alert_rules_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Alert rule not found: " + rule_id);
        }

        const auto& rule = *rule_it;

        if (!rule.enabled || !metrics_service_) {
            return true;
        }

        // Get the metric value
        auto metric_result = metrics_service_->get_metric_value(rule.metric_name);
        if (!metric_result.has_value()) {
            LOG_DEBUG(logger_, "Metric not found for rule " + rule_id + ": " + rule.metric_name);
            return true; // Not an error, metric just doesn't exist yet
        }

        double metric_value = metric_result.value();

        // Check if condition is met
        if (check_metric_condition(metric_value, rule)) {
            // Create an active alert if one doesn't already exist
            auto existing_alert = std::find_if(active_alerts_.begin(), active_alerts_.end(),
                [&rule_id](const ActiveAlert& a) { return a.rule_id == rule_id && !a.resolved; });

            if (existing_alert == active_alerts_.end()) {
                ActiveAlert alert = create_active_alert(rule, metric_value);
                active_alerts_.push_back(alert);
                LOG_WARN(logger_, "Alert triggered: " + alert.name + " (value: " + std::to_string(metric_value) + ")");

                // Send the alert
                send_alert(alert);
            }
        }

        update_rule_evaluation_time(rule_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in evaluate_alert_rule: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to evaluate alert rule: " + std::string(e.what()));
    }
}

Result<std::vector<ActiveAlert>> AlertService::get_active_alerts() const {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);
        std::vector<ActiveAlert> alerts = active_alerts_;
        LOG_DEBUG(logger_, "Retrieved " + std::to_string(alerts.size()) + " active alerts");
        return alerts;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_active_alerts: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get active alerts: " + std::string(e.what()));
    }
}

Result<std::vector<ActiveAlert>> AlertService::get_active_alerts_by_rule(const std::string& rule_id) const {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);

        std::vector<ActiveAlert> filtered;
        for (const auto& alert : active_alerts_) {
            if (alert.rule_id == rule_id) {
                filtered.push_back(alert);
            }
        }

        LOG_DEBUG(logger_, "Retrieved " + std::to_string(filtered.size()) + " active alerts for rule " + rule_id);
        return filtered;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_active_alerts_by_rule: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get active alerts by rule: " + std::string(e.what()));
    }
}

Result<std::vector<ActiveAlert>> AlertService::get_active_alerts_by_status(const std::string& status) const {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);

        std::vector<ActiveAlert> filtered;
        for (const auto& alert : active_alerts_) {
            if (alert.status == status) {
                filtered.push_back(alert);
            }
        }

        LOG_DEBUG(logger_, "Retrieved " + std::to_string(filtered.size()) + " active alerts with status " + status);
        return filtered;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_active_alerts_by_status: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get active alerts by status: " + std::string(e.what()));
    }
}

Result<bool> AlertService::resolve_alert(const std::string& alert_id) {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);

        auto it = std::find_if(active_alerts_.begin(), active_alerts_.end(),
            [&alert_id](const ActiveAlert& alert) { return alert.alert_id == alert_id; });

        if (it == active_alerts_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Alert not found: " + alert_id);
        }

        it->resolved = true;
        it->status = "resolved";
        it->end_time = std::chrono::system_clock::now();
        it->updated_at = std::chrono::system_clock::now();

        LOG_INFO(logger_, "Resolved alert: " + alert_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in resolve_alert: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to resolve alert: " + std::string(e.what()));
    }
}

Result<bool> AlertService::send_alert(const ActiveAlert& alert) {
    try {
        LOG_INFO(logger_, "Sending alert: " + alert.name + " - " + alert.message);

        // Send to configured channels
        for (const auto& channel : config_.notification_channels) {
            if (channel == "log") {
                send_alert_to_log(alert);
            } else if (channel == "webhook") {
                send_alert_to_webhook(alert);
            } else if (channel == "email") {
                send_alert_to_email(alert);
            }
        }

        LOG_DEBUG(logger_, "Alert sent successfully: " + alert.name);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in send_alert: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to send alert: " + std::string(e.what()));
    }
}

Result<std::unordered_map<std::string, int>> AlertService::get_alert_stats() const {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);

        std::unordered_map<std::string, int> stats;
        stats["total_rules"] = static_cast<int>(alert_rules_.size());
        stats["total_active_alerts"] = static_cast<int>(active_alerts_.size());

        int firing = 0, pending = 0, resolved = 0;
        for (const auto& alert : active_alerts_) {
            if (alert.status == "firing") firing++;
            else if (alert.status == "pending") pending++;
            else if (alert.status == "resolved") resolved++;
        }

        stats["firing"] = firing;
        stats["pending"] = pending;
        stats["resolved"] = resolved;

        LOG_DEBUG(logger_, "Generated alert statistics");
        return stats;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in get_alert_stats: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get alert stats: " + std::string(e.what()));
    }
}

Result<bool> AlertService::update_config(const AlertConfig& new_config) {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);

        if (new_config.evaluation_interval_seconds <= 0) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid evaluation interval");
        }

        if (new_config.max_active_alerts <= 0) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid max active alerts");
        }

        config_ = new_config;
        LOG_INFO(logger_, "Updated alert configuration");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in update_config: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to update config: " + std::string(e.what()));
    }
}

Result<bool> AlertService::silence_alerts(const std::unordered_map<std::string, std::string>& match_labels,
                                          const std::chrono::system_clock::time_point& until) {
    try {
        std::lock_guard<std::mutex> lock(alert_mutex_);

        int silenced_count = 0;
        for (auto& alert : active_alerts_) {
            if (alert.resolved) continue;

            // Check if alert labels match
            bool matches = true;
            for (const auto& match : match_labels) {
                auto it = alert.labels.find(match.first);
                if (it == alert.labels.end() || it->second != match.second) {
                    matches = false;
                    break;
                }
            }

            if (matches) {
                alert.status = "silenced";
                alert.updated_at = std::chrono::system_clock::now();
                silenced_count++;
            }
        }

        LOG_INFO(logger_, "Silenced " + std::to_string(silenced_count) + " alerts");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in silence_alerts: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to silence alerts: " + std::string(e.what()));
    }
}

std::vector<std::string> AlertService::get_notification_channels() const {
    std::lock_guard<std::mutex> lock(alert_mutex_);
    return config_.notification_channels;
}

// Private helper methods

bool AlertService::check_metric_condition(double value, const AlertRule& rule) const {
    if (rule.condition == "gt") {
        return value > rule.threshold;
    } else if (rule.condition == "lt") {
        return value < rule.threshold;
    } else if (rule.condition == "eq") {
        return value == rule.threshold;
    } else if (rule.condition == "gte") {
        return value >= rule.threshold;
    } else if (rule.condition == "lte") {
        return value <= rule.threshold;
    }
    return false;
}

std::chrono::seconds AlertService::parse_duration(const std::string& duration_str) const {
    std::regex duration_regex(R"((\d+)([smh]))");
    std::smatch match;

    if (std::regex_match(duration_str, match, duration_regex)) {
        int value = std::stoi(match[1].str());
        char unit = match[2].str()[0];

        if (unit == 's') return std::chrono::seconds(value);
        if (unit == 'm') return std::chrono::minutes(value);
        if (unit == 'h') return std::chrono::hours(value);
    }

    return std::chrono::seconds(60); // Default to 1 minute
}

bool AlertService::validate_alert_rule(const AlertRule& rule) const {
    if (rule.name.empty() || rule.metric_name.empty() || rule.condition.empty()) {
        return false;
    }

    if (rule.condition != "gt" && rule.condition != "lt" &&
        rule.condition != "eq" && rule.condition != "gte" && rule.condition != "lte") {
        return false;
    }

    return true;
}

ActiveAlert AlertService::create_active_alert(const AlertRule& rule, double value) const {
    ActiveAlert alert;
    alert.alert_id = "alert_" + rule.rule_id + "_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    alert.rule_id = rule.rule_id;
    alert.name = rule.name;
    alert.status = "firing";
    alert.start_time = std::chrono::system_clock::now();
    alert.updated_at = std::chrono::system_clock::now();
    alert.value = value;
    alert.labels = rule.labels;
    alert.message = generate_alert_message(rule, value);
    alert.resolved = false;
    return alert;
}

void AlertService::cleanup_inactive_alerts() {
    std::lock_guard<std::mutex> lock(alert_mutex_);

    // Remove resolved alerts older than 24 hours
    auto cutoff = std::chrono::system_clock::now() - std::chrono::hours(24);

    active_alerts_.erase(
        std::remove_if(active_alerts_.begin(), active_alerts_.end(),
            [cutoff](const ActiveAlert& alert) {
                return alert.resolved && alert.end_time < cutoff;
            }),
        active_alerts_.end()
    );
}

Result<bool> AlertService::send_alert_to_log(const ActiveAlert& alert) {
    LOG_WARN(logger_, "ALERT [" + alert.name + "]: " + alert.message +
            " (value: " + std::to_string(alert.value) + ")");
    return true;
}

Result<bool> AlertService::send_alert_to_webhook(const ActiveAlert& alert) {
    // Stub implementation
    LOG_DEBUG(logger_, "Would send alert to webhook: " + alert.name);
    return true;
}

Result<bool> AlertService::send_alert_to_email(const ActiveAlert& alert) {
    // Stub implementation
    LOG_DEBUG(logger_, "Would send alert to email: " + alert.name);
    return true;
}

bool AlertService::should_alert_be_firing(const AlertRule& rule, double current_value) const {
    return check_metric_condition(current_value, rule);
}

std::string AlertService::generate_alert_message(const AlertRule& rule, double value) const {
    std::stringstream ss;
    ss << rule.description << ": metric '" << rule.metric_name << "' is " << value;
    ss << " (threshold: " << rule.threshold << ", condition: " << rule.condition << ")";
    return ss.str();
}

void AlertService::update_rule_evaluation_time(const std::string& rule_id) {
    rule_last_evaluated_[rule_id] = std::chrono::system_clock::now();
}

bool AlertService::should_evaluate_rule(const std::string& rule_id) const {
    auto it = rule_last_evaluated_.find(rule_id);
    if (it == rule_last_evaluated_.end()) {
        return true; // Never evaluated before
    }

    auto elapsed = std::chrono::system_clock::now() - it->second;
    return elapsed >= std::chrono::seconds(config_.evaluation_interval_seconds);
}

} // namespace jadevectordb
