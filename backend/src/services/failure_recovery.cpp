#include "failure_recovery.h"
#include "lib/error_handling.h"
#include <chrono>
#include <algorithm>
#include <random>

namespace jadevectordb {

std::string recovery_action_to_string(RecoveryAction action) {
    switch (action) {
        case RecoveryAction::NONE: return "NONE";
        case RecoveryAction::RESTART_NODE: return "RESTART_NODE";
        case RecoveryAction::REASSIGN_SHARD: return "REASSIGN_SHARD";
        case RecoveryAction::MIGRATE_SHARD: return "MIGRATE_SHARD";
        case RecoveryAction::PROMOTE_REPLICA: return "PROMOTE_REPLICA";
        case RecoveryAction::REBUILD_INDEX: return "REBUILD_INDEX";
        case RecoveryAction::RESTORE_FROM_BACKUP: return "RESTORE_FROM_BACKUP";
        default: return "UNKNOWN";
    }
}

std::string failure_type_to_string(FailureType type) {
    switch (type) {
        case FailureType::NODE_DOWN: return "NODE_DOWN";
        case FailureType::NODE_SLOW: return "NODE_SLOW";
        case FailureType::NETWORK_PARTITION: return "NETWORK_PARTITION";
        case FailureType::DISK_FULL: return "DISK_FULL";
        case FailureType::MEMORY_EXHAUSTED: return "MEMORY_EXHAUSTED";
        case FailureType::HIGH_LATENCY: return "HIGH_LATENCY";
        case FailureType::DATA_CORRUPTION: return "DATA_CORRUPTION";
        case FailureType::UNKNOWN: return "UNKNOWN";
        default: return "UNKNOWN";
    }
}

FailureType string_to_failure_type(const std::string& str) {
    if (str == "NODE_DOWN") return FailureType::NODE_DOWN;
    if (str == "NODE_SLOW") return FailureType::NODE_SLOW;
    if (str == "NETWORK_PARTITION") return FailureType::NETWORK_PARTITION;
    if (str == "DISK_FULL") return FailureType::DISK_FULL;
    if (str == "MEMORY_EXHAUSTED") return FailureType::MEMORY_EXHAUSTED;
    if (str == "HIGH_LATENCY") return FailureType::HIGH_LATENCY;
    if (str == "DATA_CORRUPTION") return FailureType::DATA_CORRUPTION;
    return FailureType::UNKNOWN;
}

FailureRecoveryService::FailureRecoveryService() {
    logger_ = logging::LoggerManager::get_logger("FailureRecoveryService");
}

FailureRecoveryService::~FailureRecoveryService() {
    stop();
}

bool FailureRecoveryService::initialize(
    std::shared_ptr<HealthMonitor> health_monitor,
    std::shared_ptr<ShardingService> sharding_service,
    std::shared_ptr<LiveMigrationService> migration_service) {
    try {
        if (!health_monitor || !sharding_service) {
            LOG_ERROR(logger_, "Required services are null");
            return false;
        }
        
        health_monitor_ = health_monitor;
        sharding_service_ = sharding_service;
        migration_service_ = migration_service;
        
        LOG_INFO(logger_, "FailureRecoveryService initialized");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in initialize: " + std::string(e.what()));
        return false;
    }
}

Result<bool> FailureRecoveryService::start() {
    try {
        if (running_) {
            LOG_WARN(logger_, "FailureRecoveryService already running");
            return true;
        }
        
        LOG_INFO(logger_, "Starting FailureRecoveryService");
        running_ = true;
        
        monitor_thread_ = std::make_unique<std::thread>(&FailureRecoveryService::monitoring_loop, this);
        
        LOG_INFO(logger_, "FailureRecoveryService started");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in start: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to start: " + std::string(e.what()));
    }
}

void FailureRecoveryService::stop() {
    try {
        if (!running_) {
            return;
        }
        
        LOG_INFO(logger_, "Stopping FailureRecoveryService");
        running_ = false;
        
        if (monitor_thread_ && monitor_thread_->joinable()) {
            monitor_thread_->join();
        }
        monitor_thread_.reset();
        
        // Stop all test threads
        std::lock_guard<std::mutex> lock(test_threads_mutex_);
        for (auto& pair : test_threads_) {
            if (pair.second && pair.second->joinable()) {
                pair.second->join();
            }
        }
        test_threads_.clear();
        
        LOG_INFO(logger_, "FailureRecoveryService stopped");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in stop: " + std::string(e.what()));
    }
}

Result<FailureType> FailureRecoveryService::detect_failure(const std::string& node_id) {
    try {
        if (!health_monitor_) {
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "HealthMonitor not initialized");
        }
        
        auto status_result = health_monitor_->get_node_status(node_id);
        if (!status_result.has_value()) {
            RETURN_ERROR(status_result.error().code, status_result.error().message);
        }
        
        HealthStatus status = status_result.value();
        
        if (status == HealthStatus::UNHEALTHY) {
            auto health_result = health_monitor_->check_node_health(node_id);
            if (health_result.has_value()) {
                const NodeHealth& health = health_result.value();
                
                if (!health.is_reachable) {
                    return FailureType::NODE_DOWN;
                }
                if (health.disk_usage > 95.0) {
                    return FailureType::DISK_FULL;
                }
                if (health.memory_usage > 95.0) {
                    return FailureType::MEMORY_EXHAUSTED;
                }
                if (health.response_time_ms > 5000) {
                    return FailureType::HIGH_LATENCY;
                }
            }
            return FailureType::UNKNOWN;
        } else if (status == HealthStatus::DEGRADED) {
            return FailureType::NODE_SLOW;
        }
        
        return FailureType::UNKNOWN;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in detect_failure: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to detect failure: " + std::string(e.what()));
    }
}

Result<bool> FailureRecoveryService::report_failure(
    const std::string& node_id,
    FailureType type,
    const std::string& details) {
    try {
        LOG_WARN(logger_, "Failure reported for node " + node_id + ": " + 
                failure_type_to_string(type) + " - " + details);
        
        if (auto_recovery_enabled_) {
            auto recovery_result = trigger_recovery(node_id, type);
            if (!recovery_result.has_value()) {
                LOG_ERROR(logger_, "Failed to trigger recovery: " + recovery_result.error().message);
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to report failure: " + std::string(e.what()));
    }
}

Result<std::string> FailureRecoveryService::trigger_recovery(const std::string& node_id, FailureType type) {
    try {
        std::string recovery_id = generate_recovery_id();
        
        RecoveryStatus status;
        status.recovery_id = recovery_id;
        status.node_id = node_id;
        status.failure_type = type;
        status.action = determine_recovery_action(node_id, type);
        status.status = "in_progress";
        status.started_at = get_current_timestamp();
        status.completed_at = 0;
        
        {
            std::lock_guard<std::mutex> lock(recoveries_mutex_);
            active_recoveries_[recovery_id] = status;
        }
        
        LOG_INFO(logger_, "Triggered recovery " + recovery_id + " for node " + node_id + 
                " with action: " + recovery_action_to_string(status.action));
        
        // Execute recovery in background
        std::thread recovery_thread([this, recovery_id, node_id, type]() {
            execute_recovery(recovery_id, node_id, type);
        });
        recovery_thread.detach();
        
        return recovery_id;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in trigger_recovery: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to trigger recovery: " + std::string(e.what()));
    }
}

Result<bool> FailureRecoveryService::reassign_shards(const std::string& failed_node) {
    try {
        if (!sharding_service_) {
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "ShardingService not initialized");
        }
        
        LOG_INFO(logger_, "Reassigning shards from failed node: " + failed_node);
        
        // Get all registered nodes
        auto nodes = health_monitor_->get_registered_nodes();
        std::vector<std::string> healthy_nodes;
        
        for (const auto& node : nodes) {
            if (node != failed_node) {
                auto status_result = health_monitor_->get_node_status(node);
                if (status_result.has_value() && status_result.value() == HealthStatus::HEALTHY) {
                    healthy_nodes.push_back(node);
                }
            }
        }
        
        if (healthy_nodes.empty()) {
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "No healthy nodes available for reassignment");
        }
        
        LOG_INFO(logger_, "Found " + std::to_string(healthy_nodes.size()) + " healthy nodes for reassignment");
        
        // In a real implementation, we would:
        // 1. Get list of shards on failed node
        // 2. For each shard, pick a healthy node
        // 3. Initiate migration or promote replica
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in reassign_shards: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to reassign shards: " + std::string(e.what()));
    }
}

Result<bool> FailureRecoveryService::promote_replica(const std::string& shard_id, const std::string& new_primary) {
    try {
        LOG_INFO(logger_, "Promoting replica for shard " + shard_id + " on node " + new_primary);
        
        // In a real implementation:
        // 1. Verify replica is in sync
        // 2. Update shard metadata to mark new primary
        // 3. Redirect traffic to new primary
        // 4. Notify other replicas of leadership change
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to promote replica: " + std::string(e.what()));
    }
}

Result<bool> FailureRecoveryService::rebuild_data(const std::string& shard_id) {
    try {
        LOG_INFO(logger_, "Rebuilding data for shard: " + shard_id);
        
        // In a real implementation:
        // 1. Create new shard on healthy node
        // 2. Copy data from replicas or backup
        // 3. Rebuild indices
        // 4. Verify data integrity
        // 5. Update shard metadata
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to rebuild data: " + std::string(e.what()));
    }
}

Result<bool> FailureRecoveryService::handle_master_failover() {
    try {
        LOG_WARN(logger_, "Handling master failover");
        
        auto new_master_result = elect_new_master();
        if (!new_master_result.has_value()) {
            RETURN_ERROR(new_master_result.error().code, "Master election failed: " + new_master_result.error().message);
        }
        
        std::string new_master = new_master_result.value();
        LOG_INFO(logger_, "New master elected: " + new_master);
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Master failover failed: " + std::string(e.what()));
    }
}

Result<std::string> FailureRecoveryService::elect_new_master() {
    try {
        if (!health_monitor_) {
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "HealthMonitor not initialized");
        }
        
        auto nodes = health_monitor_->get_registered_nodes();
        for (const auto& node : nodes) {
            auto status_result = health_monitor_->get_node_status(node);
            if (status_result.has_value() && status_result.value() == HealthStatus::HEALTHY) {
                LOG_INFO(logger_, "Elected node " + node + " as new master");
                return node;
            }
        }
        
        RETURN_ERROR(ErrorCode::NOT_FOUND, "No healthy nodes available for master election");
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Master election failed: " + std::string(e.what()));
    }
}

Result<RecoveryStatus> FailureRecoveryService::get_recovery_status(const std::string& recovery_id) {
    try {
        std::lock_guard<std::mutex> lock(recoveries_mutex_);
        
        auto it = active_recoveries_.find(recovery_id);
        if (it != active_recoveries_.end()) {
            return it->second;
        }
        
        // Check history
        auto hist_it = std::find_if(recovery_history_.begin(), recovery_history_.end(),
            [&recovery_id](const RecoveryStatus& s) { return s.recovery_id == recovery_id; });
        
        if (hist_it != recovery_history_.end()) {
            return *hist_it;
        }
        
        RETURN_ERROR(ErrorCode::NOT_FOUND, "Recovery not found: " + recovery_id);
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get recovery status: " + std::string(e.what()));
    }
}

std::vector<RecoveryStatus> FailureRecoveryService::get_active_recoveries() {
    std::lock_guard<std::mutex> lock(recoveries_mutex_);
    std::vector<RecoveryStatus> active;
    for (const auto& pair : active_recoveries_) {
        active.push_back(pair.second);
    }
    return active;
}

std::vector<RecoveryStatus> FailureRecoveryService::get_recovery_history(int limit) {
    std::lock_guard<std::mutex> lock(recoveries_mutex_);
    if (limit <= 0 || limit > static_cast<int>(recovery_history_.size())) {
        return recovery_history_;
    }
    return std::vector<RecoveryStatus>(recovery_history_.end() - limit, recovery_history_.end());
}

void FailureRecoveryService::register_recovery_callback(RecoveryCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    recovery_callbacks_.push_back(callback);
}

Result<std::string> FailureRecoveryService::run_chaos_test(const ChaosTestConfig& config) {
    try {
        std::string test_id = generate_test_id();
        
        ChaosTestResult result;
        result.test_id = test_id;
        result.test_name = config.test_name;
        result.failure_type = config.failure_type;
        result.started_at = get_current_timestamp();
        result.completed_at = 0;
        result.success = false;
        result.failures_injected = 0;
        result.recoveries_attempted = 0;
        result.recoveries_successful = 0;
        result.avg_recovery_time_seconds = 0.0;
        
        {
            std::lock_guard<std::mutex> lock(chaos_mutex_);
            chaos_tests_[test_id] = result;
        }
        
        LOG_INFO(logger_, "Starting chaos test: " + config.test_name + " (ID: " + test_id + ")");
        
        // Execute test in background
        {
            std::lock_guard<std::mutex> lock(test_threads_mutex_);
            test_threads_[test_id] = std::make_unique<std::thread>(
                &FailureRecoveryService::execute_chaos_test, this, test_id, config);
        }
        
        return test_id;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in run_chaos_test: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to run chaos test: " + std::string(e.what()));
    }
}

Result<ChaosTestResult> FailureRecoveryService::get_chaos_test_result(const std::string& test_id) {
    try {
        std::lock_guard<std::mutex> lock(chaos_mutex_);
        
        auto it = chaos_tests_.find(test_id);
        if (it == chaos_tests_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Chaos test not found: " + test_id);
        }
        
        return it->second;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get chaos test result: " + std::string(e.what()));
    }
}

std::vector<ChaosTestResult> FailureRecoveryService::get_chaos_test_history() {
    std::lock_guard<std::mutex> lock(chaos_mutex_);
    std::vector<ChaosTestResult> history;
    for (const auto& pair : chaos_tests_) {
        history.push_back(pair.second);
    }
    return history;
}

Result<bool> FailureRecoveryService::inject_node_failure(const std::string& node_id, int duration_seconds) {
    try {
        LOG_WARN(logger_, "Injecting node failure for " + node_id + " for " + std::to_string(duration_seconds) + "s");
        
        // Simulate node failure
        if (health_monitor_) {
            NodeHealth health;
            health.node_id = node_id;
            health.status = HealthStatus::UNHEALTHY;
            health.is_reachable = false;
            health.error_message = "Chaos: Injected node failure";
            health_monitor_->update_node_health(node_id, health);
        }
        
        // Schedule recovery after duration
        std::thread([this, node_id, duration_seconds]() {
            std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
            
            if (health_monitor_) {
                NodeHealth health;
                health.node_id = node_id;
                health.status = HealthStatus::HEALTHY;
                health.is_reachable = true;
                health.cpu_usage = 50.0;
                health.memory_usage = 60.0;
                health.disk_usage = 40.0;
                health_monitor_->update_node_health(node_id, health);
                
                LOG_INFO(logger_, "Chaos: Restored node " + node_id + " after failure injection");
            }
        }).detach();
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to inject node failure: " + std::string(e.what()));
    }
}

Result<bool> FailureRecoveryService::inject_network_partition(
    const std::vector<std::string>& nodes,
    int duration_seconds) {
    try {
        LOG_WARN(logger_, "Injecting network partition for " + std::to_string(nodes.size()) + 
                " nodes for " + std::to_string(duration_seconds) + "s");
        
        // Mark nodes as unreachable from each other
        for (const auto& node : nodes) {
            if (health_monitor_) {
                NodeHealth health;
                health.node_id = node;
                health.status = HealthStatus::DEGRADED;
                health.is_reachable = false;
                health.error_message = "Chaos: Network partition";
                health_monitor_->update_node_health(node, health);
            }
        }
        
        // Schedule recovery
        std::thread([this, nodes, duration_seconds]() {
            std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
            
            for (const auto& node : nodes) {
                if (health_monitor_) {
                    health_monitor_->record_heartbeat(node);
                }
            }
            
            LOG_INFO(logger_, "Chaos: Restored network partition");
        }).detach();
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to inject network partition: " + std::string(e.what()));
    }
}

Result<bool> FailureRecoveryService::inject_high_latency(
    const std::string& node_id,
    int latency_ms,
    int duration_seconds) {
    try {
        LOG_WARN(logger_, "Injecting high latency (" + std::to_string(latency_ms) + 
                "ms) for node " + node_id);
        
        if (health_monitor_) {
            NodeHealth health;
            health.node_id = node_id;
            health.status = HealthStatus::DEGRADED;
            health.response_time_ms = latency_ms;
            health.error_message = "Chaos: High latency";
            health_monitor_->update_node_health(node_id, health);
        }
        
        std::thread([this, node_id, duration_seconds]() {
            std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
            
            if (health_monitor_) {
                health_monitor_->record_heartbeat(node_id);
            }
        }).detach();
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to inject high latency: " + std::string(e.what()));
    }
}

Result<bool> FailureRecoveryService::inject_resource_exhaustion(
    const std::string& node_id,
    const std::string& resource) {
    try {
        LOG_WARN(logger_, "Injecting resource exhaustion (" + resource + ") for node " + node_id);
        
        if (health_monitor_) {
            NodeHealth health;
            health.node_id = node_id;
            health.status = HealthStatus::UNHEALTHY;
            health.is_reachable = true;
            
            if (resource == "memory") {
                health.memory_usage = 98.0;
            } else if (resource == "disk") {
                health.disk_usage = 98.0;
            } else if (resource == "cpu") {
                health.cpu_usage = 98.0;
            }
            
            health.error_message = "Chaos: " + resource + " exhaustion";
            health_monitor_->update_node_health(node_id, health);
        }
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to inject resource exhaustion: " + std::string(e.what()));
    }
}

// Private methods

void FailureRecoveryService::monitoring_loop() {
    LOG_INFO(logger_, "Failure recovery monitoring loop started");
    
    while (running_) {
        try {
            check_for_failures();
            process_recovery_queue();
            
            std::this_thread::sleep_for(std::chrono::seconds(10));
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Exception in monitoring_loop: " + std::string(e.what()));
        }
    }
    
    LOG_INFO(logger_, "Failure recovery monitoring loop stopped");
}

void FailureRecoveryService::check_for_failures() {
    if (!health_monitor_ || !auto_recovery_enabled_) {
        return;
    }
    
    auto nodes = health_monitor_->get_registered_nodes();
    for (const auto& node : nodes) {
        auto failure_result = detect_failure(node);
        if (failure_result.has_value() && failure_result.value() != FailureType::UNKNOWN) {
            LOG_WARN(logger_, "Detected failure on node " + node + ": " + 
                    failure_type_to_string(failure_result.value()));
            
            // Check if recovery already in progress
            bool recovery_in_progress = false;
            {
                std::lock_guard<std::mutex> lock(recoveries_mutex_);
                for (const auto& pair : active_recoveries_) {
                    if (pair.second.node_id == node && pair.second.status == "in_progress") {
                        recovery_in_progress = true;
                        break;
                    }
                }
            }
            
            if (!recovery_in_progress) {
                trigger_recovery(node, failure_result.value());
            }
        }
    }
}

void FailureRecoveryService::process_recovery_queue() {
    std::vector<std::string> to_remove;
    
    {
        std::lock_guard<std::mutex> lock(recoveries_mutex_);
        
        int64_t now = get_current_timestamp();
        for (auto& pair : active_recoveries_) {
            RecoveryStatus& status = pair.second;
            
            // Check for timeout
            if (status.status == "in_progress" && 
                (now - status.started_at) > recovery_timeout_seconds_) {
                status.status = "failed";
                status.error_message = "Recovery timeout";
                status.completed_at = now;
                to_remove.push_back(pair.first);
            }
            
            // Move completed to history
            if (status.status == "completed" || status.status == "failed") {
                recovery_history_.push_back(status);
                to_remove.push_back(pair.first);
                
                // Keep only last 1000 in history
                if (recovery_history_.size() > 1000) {
                    recovery_history_.erase(recovery_history_.begin());
                }
            }
        }
        
        for (const auto& id : to_remove) {
            active_recoveries_.erase(id);
        }
    }
}

void FailureRecoveryService::execute_recovery(
    const std::string& recovery_id,
    const std::string& node_id,
    FailureType type) {
    try {
        LOG_INFO(logger_, "Executing recovery " + recovery_id + " for node " + node_id);
        
        RecoveryAction action = determine_recovery_action(node_id, type);
        bool success = false;
        
        switch (action) {
            case RecoveryAction::RESTART_NODE:
                success = execute_node_restart(node_id);
                break;
            case RecoveryAction::REASSIGN_SHARD:
                success = execute_shard_reassignment(node_id);
                break;
            case RecoveryAction::PROMOTE_REPLICA:
                success = execute_replica_promotion("");
                break;
            default:
                LOG_WARN(logger_, "No recovery action determined");
                break;
        }
        
        // Update recovery status
        {
            std::lock_guard<std::mutex> lock(recoveries_mutex_);
            auto it = active_recoveries_.find(recovery_id);
            if (it != active_recoveries_.end()) {
                it->second.status = success ? "completed" : "failed";
                it->second.completed_at = get_current_timestamp();
                if (!success) {
                    it->second.error_message = "Recovery action failed";
                }
                notify_recovery_status(it->second);
            }
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in execute_recovery: " + std::string(e.what()));
        
        std::lock_guard<std::mutex> lock(recoveries_mutex_);
        auto it = active_recoveries_.find(recovery_id);
        if (it != active_recoveries_.end()) {
            it->second.status = "failed";
            it->second.error_message = std::string(e.what());
            it->second.completed_at = get_current_timestamp();
        }
    }
}

RecoveryAction FailureRecoveryService::determine_recovery_action(
    const std::string& node_id,
    FailureType type) {
    switch (type) {
        case FailureType::NODE_DOWN:
            return RecoveryAction::REASSIGN_SHARD;
        case FailureType::NODE_SLOW:
            return RecoveryAction::RESTART_NODE;
        case FailureType::DISK_FULL:
        case FailureType::MEMORY_EXHAUSTED:
            return RecoveryAction::MIGRATE_SHARD;
        case FailureType::DATA_CORRUPTION:
            return RecoveryAction::REBUILD_INDEX;
        default:
            return RecoveryAction::NONE;
    }
}

bool FailureRecoveryService::execute_node_restart(const std::string& node_id) {
    LOG_INFO(logger_, "Restarting node: " + node_id);
    // In production, this would trigger actual node restart
    return true;
}

bool FailureRecoveryService::execute_shard_reassignment(const std::string& node_id) {
    LOG_INFO(logger_, "Reassigning shards from node: " + node_id);
    auto result = reassign_shards(node_id);
    return result.has_value();
}

bool FailureRecoveryService::execute_shard_migration(
    const std::string& shard_id,
    const std::string& target_node) {
    if (!migration_service_) {
        return false;
    }
    
    LOG_INFO(logger_, "Migrating shard " + shard_id + " to " + target_node);
    auto plan_result = migration_service_->create_migration_plan(
        shard_id, target_node, MigrationStrategy::LIVE_MIGRATION);
    
    if (plan_result.has_value()) {
        auto start_result = migration_service_->start_migration(plan_result.value());
        return start_result.has_value();
    }
    
    return false;
}

bool FailureRecoveryService::execute_replica_promotion(const std::string& shard_id) {
    LOG_INFO(logger_, "Promoting replica for shard: " + shard_id);
    // Would select a healthy replica and promote it
    return true;
}

std::string FailureRecoveryService::generate_recovery_id() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return "rec_" + std::to_string(duration.count());
}

std::string FailureRecoveryService::generate_test_id() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return "chaos_" + std::to_string(duration.count());
}

int64_t FailureRecoveryService::get_current_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void FailureRecoveryService::notify_recovery_status(const RecoveryStatus& status) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    for (const auto& callback : recovery_callbacks_) {
        try {
            callback(status);
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Exception in recovery callback: " + std::string(e.what()));
        }
    }
}

void FailureRecoveryService::execute_chaos_test(const std::string& test_id, const ChaosTestConfig& config) {
    try {
        LOG_INFO(logger_, "Executing chaos test: " + config.test_name);
        
        int64_t start_time = get_current_timestamp();
        std::vector<int64_t> recovery_times;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        int failures_injected = 0;
        int recoveries_successful = 0;
        
        // Inject failures based on configuration
        for (const auto& node : config.target_nodes) {
            if (dis(gen) < config.failure_probability) {
                auto inject_result = inject_node_failure(node, config.duration_seconds / 2);
                if (inject_result.has_value()) {
                    failures_injected++;
                    
                    if (config.auto_recovery_enabled) {
                        int64_t recovery_start = get_current_timestamp();
                        auto recovery_result = trigger_recovery(node, config.failure_type);
                        
                        if (recovery_result.has_value()) {
                            // Wait for recovery to complete
                            std::this_thread::sleep_for(std::chrono::seconds(5));
                            
                            auto status_result = get_recovery_status(recovery_result.value());
                            if (status_result.has_value() && 
                                status_result.value().status == "completed") {
                                recoveries_successful++;
                                int64_t recovery_time = get_current_timestamp() - recovery_start;
                                recovery_times.push_back(recovery_time);
                            }
                        }
                    }
                }
            }
        }
        
        // Wait for test duration
        std::this_thread::sleep_for(std::chrono::seconds(config.duration_seconds));
        
        // Calculate results
        double avg_recovery_time = 0.0;
        if (!recovery_times.empty()) {
            int64_t total = 0;
            for (auto t : recovery_times) {
                total += t;
            }
            avg_recovery_time = static_cast<double>(total) / recovery_times.size();
        }
        
        // Update test result
        {
            std::lock_guard<std::mutex> lock(chaos_mutex_);
            auto it = chaos_tests_.find(test_id);
            if (it != chaos_tests_.end()) {
                it->second.completed_at = get_current_timestamp();
                it->second.success = (recoveries_successful >= failures_injected / 2);
                it->second.failures_injected = failures_injected;
                it->second.recoveries_attempted = failures_injected;
                it->second.recoveries_successful = recoveries_successful;
                it->second.avg_recovery_time_seconds = avg_recovery_time;
            }
        }
        
        LOG_INFO(logger_, "Chaos test " + test_id + " completed: " + 
                std::to_string(recoveries_successful) + "/" + 
                std::to_string(failures_injected) + " recoveries successful");
        
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in execute_chaos_test: " + std::string(e.what()));
        
        std::lock_guard<std::mutex> lock(chaos_mutex_);
        auto it = chaos_tests_.find(test_id);
        if (it != chaos_tests_.end()) {
            it->second.completed_at = get_current_timestamp();
            it->second.success = false;
            it->second.errors.push_back(std::string(e.what()));
        }
    }
}

} // namespace jadevectordb
