#pragma once

#include "lib/result.h"
#include "lib/logging.h"
#include "health_monitor.h"
#include "sharding_service.h"
#include "live_migration_service.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>

namespace jadevectordb {

// Recovery action types
enum class RecoveryAction {
    NONE,
    RESTART_NODE,
    REASSIGN_SHARD,
    MIGRATE_SHARD,
    PROMOTE_REPLICA,
    REBUILD_INDEX,
    RESTORE_FROM_BACKUP
};

// Failure types
enum class FailureType {
    NODE_DOWN,
    NODE_SLOW,
    NETWORK_PARTITION,
    DISK_FULL,
    MEMORY_EXHAUSTED,
    HIGH_LATENCY,
    DATA_CORRUPTION,
    UNKNOWN
};

// Recovery status
struct RecoveryStatus {
    std::string recovery_id;
    std::string node_id;
    std::string shard_id;
    FailureType failure_type;
    RecoveryAction action;
    std::string status; // "in_progress", "completed", "failed"
    int64_t started_at;
    int64_t completed_at;
    std::string error_message;
    std::map<std::string, std::string> metadata;
};

// Chaos test configuration
struct ChaosTestConfig {
    std::string test_name;
    FailureType failure_type;
    std::vector<std::string> target_nodes;
    int duration_seconds;
    double failure_probability;  // 0.0 to 1.0
    bool auto_recovery_enabled;
    int recovery_timeout_seconds;
};

// Chaos test result
struct ChaosTestResult {
    std::string test_id;
    std::string test_name;
    FailureType failure_type;
    int64_t started_at;
    int64_t completed_at;
    bool success;
    int failures_injected;
    int recoveries_attempted;
    int recoveries_successful;
    double avg_recovery_time_seconds;
    std::vector<std::string> errors;
    std::map<std::string, std::string> metrics;
};

// Recovery callback
using RecoveryCallback = std::function<void(const RecoveryStatus&)>;

class FailureRecoveryService {
public:
    FailureRecoveryService();
    ~FailureRecoveryService();
    
    // Initialization
    bool initialize(std::shared_ptr<HealthMonitor> health_monitor,
                   std::shared_ptr<ShardingService> sharding_service,
                   std::shared_ptr<LiveMigrationService> migration_service);
    Result<bool> start();
    void stop();
    bool is_running() const { return running_; }
    
    // Failure detection
    Result<FailureType> detect_failure(const std::string& node_id);
    Result<bool> report_failure(const std::string& node_id, FailureType type, const std::string& details);
    
    // Recovery actions
    Result<std::string> trigger_recovery(const std::string& node_id, FailureType type);
    Result<bool> reassign_shards(const std::string& failed_node);
    Result<bool> promote_replica(const std::string& shard_id, const std::string& new_primary);
    Result<bool> rebuild_data(const std::string& shard_id);
    
    // Master failover
    Result<bool> handle_master_failover();
    Result<std::string> elect_new_master();
    
    // Recovery status
    Result<RecoveryStatus> get_recovery_status(const std::string& recovery_id);
    std::vector<RecoveryStatus> get_active_recoveries();
    std::vector<RecoveryStatus> get_recovery_history(int limit = 100);
    
    // Configuration
    void set_auto_recovery_enabled(bool enabled) { auto_recovery_enabled_ = enabled; }
    void set_recovery_timeout(int seconds) { recovery_timeout_seconds_ = seconds; }
    void register_recovery_callback(RecoveryCallback callback);
    
    // Chaos testing
    Result<std::string> run_chaos_test(const ChaosTestConfig& config);
    Result<ChaosTestResult> get_chaos_test_result(const std::string& test_id);
    std::vector<ChaosTestResult> get_chaos_test_history();
    
    // Chaos test scenarios
    Result<bool> inject_node_failure(const std::string& node_id, int duration_seconds);
    Result<bool> inject_network_partition(const std::vector<std::string>& nodes, int duration_seconds);
    Result<bool> inject_high_latency(const std::string& node_id, int latency_ms, int duration_seconds);
    Result<bool> inject_resource_exhaustion(const std::string& node_id, const std::string& resource);
    
private:
    // Monitoring thread
    void monitoring_loop();
    void check_for_failures();
    void process_recovery_queue();
    
    // Recovery execution
    void execute_recovery(const std::string& recovery_id, const std::string& node_id, FailureType type);
    RecoveryAction determine_recovery_action(const std::string& node_id, FailureType type);
    bool execute_node_restart(const std::string& node_id);
    bool execute_shard_reassignment(const std::string& node_id);
    bool execute_shard_migration(const std::string& shard_id, const std::string& target_node);
    bool execute_replica_promotion(const std::string& shard_id);
    
    // Helper methods
    std::string generate_recovery_id();
    std::string generate_test_id();
    int64_t get_current_timestamp();
    void notify_recovery_status(const RecoveryStatus& status);
    
    // Chaos testing helpers
    void execute_chaos_test(const std::string& test_id, const ChaosTestConfig& config);
    bool simulate_failure(const std::string& node_id, FailureType type, int duration_seconds);
    void record_chaos_metrics(const std::string& test_id, const std::string& metric, const std::string& value);
    
    // Dependencies
    std::shared_ptr<HealthMonitor> health_monitor_;
    std::shared_ptr<ShardingService> sharding_service_;
    std::shared_ptr<LiveMigrationService> migration_service_;
    
    // Recovery tracking
    std::map<std::string, RecoveryStatus> active_recoveries_;
    std::vector<RecoveryStatus> recovery_history_;
    mutable std::mutex recoveries_mutex_;
    
    // Chaos testing
    std::map<std::string, ChaosTestResult> chaos_tests_;
    mutable std::mutex chaos_mutex_;
    
    std::map<std::string, std::unique_ptr<std::thread>> test_threads_;
    mutable std::mutex test_threads_mutex_;
    
    // Callbacks
    std::vector<RecoveryCallback> recovery_callbacks_;
    mutable std::mutex callbacks_mutex_;
    
    // Configuration
    bool auto_recovery_enabled_ = true;
    int recovery_timeout_seconds_ = 300; // 5 minutes
    
    // Monitoring thread
    std::unique_ptr<std::thread> monitor_thread_;
    std::atomic<bool> running_{false};
    
    // Logger
    std::shared_ptr<logging::Logger> logger_;
};

// Helper functions
std::string recovery_action_to_string(RecoveryAction action);
std::string failure_type_to_string(FailureType type);
FailureType string_to_failure_type(const std::string& str);

} // namespace jadevectordb
