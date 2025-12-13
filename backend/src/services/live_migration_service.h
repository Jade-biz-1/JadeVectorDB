#pragma once

#include "lib/result.h"
#include "lib/logging.h"
#include "sharding_service.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>

namespace jadevectordb {

// Migration strategies
enum class MigrationStrategy {
    STOP_AND_COPY,      // Stop writes, copy data, resume
    LIVE_MIGRATION,     // Copy data while serving traffic
    DOUBLE_WRITE,       // Write to both old and new locations
    STAGED_MIGRATION    // Migrate in chunks over time
};

// Migration phase
enum class MigrationPhase {
    PLANNING,
    PREPARING,
    COPYING,
    SYNCING,
    SWITCHING,
    VERIFYING,
    COMPLETED,
    FAILED,
    ROLLING_BACK
};

// Enhanced migration status
struct LiveMigrationStatus {
    std::string migration_id;
    std::string shard_id;
    std::string source_node;
    std::string target_node;
    MigrationStrategy strategy;
    MigrationPhase phase;
    
    // Progress tracking
    int64_t total_vectors;
    int64_t copied_vectors;
    int64_t synced_vectors;
    int64_t verified_vectors;
    int64_t total_bytes;
    int64_t transferred_bytes;
    
    // Timing
    int64_t started_at;
    int64_t phase_started_at;
    int64_t completed_at;
    int64_t estimated_completion;
    
    // Performance metrics
    double transfer_rate_mbps;
    double avg_latency_ms;
    int64_t checkpoint_count;
    
    // Error tracking
    std::string error_message;
    int retry_count;
    bool can_rollback;
    
    // Zero-downtime metrics
    int64_t read_redirects;
    int64_t write_redirects;
    int64_t double_writes;
};

// Migration checkpoint for rollback
struct MigrationCheckpoint {
    std::string checkpoint_id;
    std::string migration_id;
    MigrationPhase phase;
    int64_t vectors_copied;
    int64_t timestamp;
    std::map<std::string, std::string> metadata;
};

// Migration plan
struct MigrationPlan {
    std::string shard_id;
    std::string source_node;
    std::string target_node;
    MigrationStrategy strategy;
    int64_t estimated_duration_seconds;
    int64_t estimated_data_size;
    int chunk_size;
    int parallel_streams;
    bool verify_data;
    bool allow_rollback;
};

// Callback for migration progress
using MigrationProgressCallback = std::function<void(const LiveMigrationStatus&)>;

class LiveMigrationService {
public:
    LiveMigrationService();
    ~LiveMigrationService();
    
    // Initialization
    bool initialize(std::shared_ptr<ShardingService> sharding_service);
    Result<bool> start();
    void stop();
    bool is_running() const { return running_; }
    
    // Migration planning
    Result<MigrationPlan> create_migration_plan(const std::string& shard_id,
                                               const std::string& target_node,
                                               MigrationStrategy strategy);
    Result<bool> validate_migration_plan(const MigrationPlan& plan);
    
    // Migration execution
    Result<std::string> start_migration(const MigrationPlan& plan);
    Result<bool> pause_migration(const std::string& migration_id);
    Result<bool> resume_migration(const std::string& migration_id);
    Result<bool> cancel_migration(const std::string& migration_id);
    
    // Rollback support
    Result<bool> create_checkpoint(const std::string& migration_id);
    Result<bool> rollback_to_checkpoint(const std::string& migration_id, 
                                       const std::string& checkpoint_id);
    Result<bool> rollback_migration(const std::string& migration_id);
    
    // Status and monitoring
    Result<LiveMigrationStatus> get_migration_status(const std::string& migration_id);
    std::vector<LiveMigrationStatus> get_active_migrations();
    std::vector<MigrationCheckpoint> get_checkpoints(const std::string& migration_id);
    
    // Progress callbacks
    void register_progress_callback(MigrationProgressCallback callback);
    
    // Zero-downtime support
    Result<bool> enable_double_write(const std::string& shard_id);
    Result<bool> disable_double_write(const std::string& shard_id);
    Result<bool> redirect_reads(const std::string& shard_id, const std::string& target_node);
    Result<bool> redirect_writes(const std::string& shard_id, const std::string& target_node);
    
private:
    // Migration execution methods
    void execute_live_migration(const std::string& migration_id, const MigrationPlan& plan);
    void execute_stop_and_copy(const std::string& migration_id, const MigrationPlan& plan);
    void execute_double_write_migration(const std::string& migration_id, const MigrationPlan& plan);
    void execute_staged_migration(const std::string& migration_id, const MigrationPlan& plan);
    
    // Migration phases
    Result<bool> phase_planning(const std::string& migration_id, const MigrationPlan& plan);
    Result<bool> phase_preparing(const std::string& migration_id, const MigrationPlan& plan);
    Result<bool> phase_copying(const std::string& migration_id, const MigrationPlan& plan);
    Result<bool> phase_syncing(const std::string& migration_id, const MigrationPlan& plan);
    Result<bool> phase_switching(const std::string& migration_id, const MigrationPlan& plan);
    Result<bool> phase_verifying(const std::string& migration_id, const MigrationPlan& plan);
    
    // Helper methods
    void update_migration_phase(const std::string& migration_id, MigrationPhase phase);
    void update_migration_progress(const std::string& migration_id, int64_t copied, int64_t synced);
    void mark_migration_failed(const std::string& migration_id, const std::string& error);
    void mark_migration_completed(const std::string& migration_id);
    
    std::string generate_migration_id();
    std::string generate_checkpoint_id();
    int64_t estimate_migration_duration(const MigrationPlan& plan);
    double calculate_transfer_rate(const LiveMigrationStatus& status);
    
    // Data transfer methods
    Result<int64_t> copy_data_chunk(const std::string& shard_id, 
                                     const std::string& target_node,
                                     int64_t offset, int chunk_size);
    Result<int64_t> sync_incremental_changes(const std::string& shard_id,
                                            const std::string& target_node,
                                            int64_t since_timestamp);
    Result<bool> verify_data_integrity(const std::string& shard_id,
                                       const std::string& source_node,
                                       const std::string& target_node);
    
    // Callback notifications
    void notify_progress(const std::string& migration_id);
    
    // Dependencies
    std::shared_ptr<ShardingService> sharding_service_;
    
    // Migration tracking
    std::map<std::string, LiveMigrationStatus> migrations_;
    mutable std::mutex migrations_mutex_;
    
    std::map<std::string, std::vector<MigrationCheckpoint>> checkpoints_;
    mutable std::mutex checkpoints_mutex_;
    
    std::map<std::string, std::unique_ptr<std::thread>> migration_threads_;
    mutable std::mutex threads_mutex_;
    
    // Progress callbacks
    std::vector<MigrationProgressCallback> progress_callbacks_;
    mutable std::mutex callbacks_mutex_;
    
    // Zero-downtime tracking
    std::map<std::string, bool> double_write_enabled_;
    std::map<std::string, std::string> read_redirections_;
    std::map<std::string, std::string> write_redirections_;
    mutable std::mutex redirections_mutex_;
    
    // Service state
    std::atomic<bool> running_{false};
    
    // Logger
    std::shared_ptr<logging::Logger> logger_;
};

// Helper functions
std::string migration_strategy_to_string(MigrationStrategy strategy);
std::string migration_phase_to_string(MigrationPhase phase);
MigrationStrategy string_to_migration_strategy(const std::string& str);

} // namespace jadevectordb
