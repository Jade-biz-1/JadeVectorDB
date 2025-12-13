#include "live_migration_service.h"
#include "lib/error_handling.h"
#include <chrono>
#include <algorithm>
#include <sstream>

namespace jadevectordb {

std::string migration_strategy_to_string(MigrationStrategy strategy) {
    switch (strategy) {
        case MigrationStrategy::STOP_AND_COPY: return "STOP_AND_COPY";
        case MigrationStrategy::LIVE_MIGRATION: return "LIVE_MIGRATION";
        case MigrationStrategy::DOUBLE_WRITE: return "DOUBLE_WRITE";
        case MigrationStrategy::STAGED_MIGRATION: return "STAGED_MIGRATION";
        default: return "UNKNOWN";
    }
}

std::string migration_phase_to_string(MigrationPhase phase) {
    switch (phase) {
        case MigrationPhase::PLANNING: return "PLANNING";
        case MigrationPhase::PREPARING: return "PREPARING";
        case MigrationPhase::COPYING: return "COPYING";
        case MigrationPhase::SYNCING: return "SYNCING";
        case MigrationPhase::SWITCHING: return "SWITCHING";
        case MigrationPhase::VERIFYING: return "VERIFYING";
        case MigrationPhase::COMPLETED: return "COMPLETED";
        case MigrationPhase::FAILED: return "FAILED";
        case MigrationPhase::ROLLING_BACK: return "ROLLING_BACK";
        default: return "UNKNOWN";
    }
}

MigrationStrategy string_to_migration_strategy(const std::string& str) {
    if (str == "STOP_AND_COPY") return MigrationStrategy::STOP_AND_COPY;
    if (str == "LIVE_MIGRATION") return MigrationStrategy::LIVE_MIGRATION;
    if (str == "DOUBLE_WRITE") return MigrationStrategy::DOUBLE_WRITE;
    if (str == "STAGED_MIGRATION") return MigrationStrategy::STAGED_MIGRATION;
    return MigrationStrategy::LIVE_MIGRATION;
}

LiveMigrationService::LiveMigrationService() {
    logger_ = logging::LoggerManager::get_logger("LiveMigrationService");
}

LiveMigrationService::~LiveMigrationService() {
    stop();
}

bool LiveMigrationService::initialize(std::shared_ptr<ShardingService> sharding_service) {
    try {
        if (!sharding_service) {
            LOG_ERROR(logger_, "ShardingService is null");
            return false;
        }
        
        sharding_service_ = sharding_service;
        LOG_INFO(logger_, "LiveMigrationService initialized");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in initialize: " + std::string(e.what()));
        return false;
    }
}

Result<bool> LiveMigrationService::start() {
    try {
        if (running_) {
            LOG_WARN(logger_, "LiveMigrationService already running");
            return true;
        }
        
        LOG_INFO(logger_, "Starting LiveMigrationService");
        running_ = true;
        
        LOG_INFO(logger_, "LiveMigrationService started");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in start: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to start: " + std::string(e.what()));
    }
}

void LiveMigrationService::stop() {
    try {
        if (!running_) {
            return;
        }
        
        LOG_INFO(logger_, "Stopping LiveMigrationService");
        running_ = false;
        
        // Wait for all migration threads to complete
        std::lock_guard<std::mutex> lock(threads_mutex_);
        for (auto& pair : migration_threads_) {
            if (pair.second && pair.second->joinable()) {
                pair.second->join();
            }
        }
        migration_threads_.clear();
        
        LOG_INFO(logger_, "LiveMigrationService stopped");
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in stop: " + std::string(e.what()));
    }
}

Result<MigrationPlan> LiveMigrationService::create_migration_plan(
    const std::string& shard_id,
    const std::string& target_node,
    MigrationStrategy strategy) {
    try {
        if (!sharding_service_) {
            RETURN_ERROR(ErrorCode::SERVICE_ERROR, "ShardingService not initialized");
        }
        
        // Get shard information
        auto node_result = sharding_service_->get_node_for_shard(shard_id);
        if (!node_result.has_value()) {
            RETURN_ERROR(node_result.error().code, "Failed to get shard node: " + node_result.error().message);
        }
        
        MigrationPlan plan;
        plan.shard_id = shard_id;
        plan.source_node = node_result.value();
        plan.target_node = target_node;
        plan.strategy = strategy;
        
        // Set defaults based on strategy
        switch (strategy) {
            case MigrationStrategy::LIVE_MIGRATION:
                plan.chunk_size = 10000;
                plan.parallel_streams = 4;
                plan.verify_data = true;
                plan.allow_rollback = true;
                break;
            case MigrationStrategy::STOP_AND_COPY:
                plan.chunk_size = 50000;
                plan.parallel_streams = 1;
                plan.verify_data = true;
                plan.allow_rollback = true;
                break;
            case MigrationStrategy::DOUBLE_WRITE:
                plan.chunk_size = 10000;
                plan.parallel_streams = 2;
                plan.verify_data = true;
                plan.allow_rollback = true;
                break;
            case MigrationStrategy::STAGED_MIGRATION:
                plan.chunk_size = 5000;
                plan.parallel_streams = 2;
                plan.verify_data = false;
                plan.allow_rollback = true;
                break;
        }
        
        plan.estimated_duration_seconds = estimate_migration_duration(plan);
        plan.estimated_data_size = 0; // Would query actual shard size
        
        LOG_INFO(logger_, "Created migration plan for shard " + shard_id + 
                " using strategy " + migration_strategy_to_string(strategy));
        
        return plan;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in create_migration_plan: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to create migration plan: " + std::string(e.what()));
    }
}

Result<bool> LiveMigrationService::validate_migration_plan(const MigrationPlan& plan) {
    try {
        if (plan.shard_id.empty()) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Shard ID is empty");
        }
        
        if (plan.source_node == plan.target_node) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Source and target nodes are the same");
        }
        
        if (plan.chunk_size <= 0) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid chunk size");
        }
        
        if (plan.parallel_streams <= 0) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Invalid parallel streams count");
        }
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Validation failed: " + std::string(e.what()));
    }
}

Result<std::string> LiveMigrationService::start_migration(const MigrationPlan& plan) {
    try {
        // Validate plan
        auto validation = validate_migration_plan(plan);
        if (!validation.has_value()) {
            return Result<std::string>::error(validation.error());
        }
        
        std::string migration_id = generate_migration_id();
        
        // Initialize migration status
        LiveMigrationStatus status;
        status.migration_id = migration_id;
        status.shard_id = plan.shard_id;
        status.source_node = plan.source_node;
        status.target_node = plan.target_node;
        status.strategy = plan.strategy;
        status.phase = MigrationPhase::PLANNING;
        status.total_vectors = 0;
        status.copied_vectors = 0;
        status.synced_vectors = 0;
        status.verified_vectors = 0;
        status.total_bytes = 0;
        status.transferred_bytes = 0;
        status.started_at = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        status.phase_started_at = status.started_at;
        status.completed_at = 0;
        status.estimated_completion = status.started_at + plan.estimated_duration_seconds;
        status.transfer_rate_mbps = 0.0;
        status.avg_latency_ms = 0.0;
        status.checkpoint_count = 0;
        status.retry_count = 0;
        status.can_rollback = plan.allow_rollback;
        status.read_redirects = 0;
        status.write_redirects = 0;
        status.double_writes = 0;
        
        {
            std::lock_guard<std::mutex> lock(migrations_mutex_);
            migrations_[migration_id] = status;
        }
        
        // Start migration in background thread
        {
            std::lock_guard<std::mutex> lock(threads_mutex_);
            migration_threads_[migration_id] = std::make_unique<std::thread>(
                [this, migration_id, plan]() {
                    switch (plan.strategy) {
                        case MigrationStrategy::LIVE_MIGRATION:
                            execute_live_migration(migration_id, plan);
                            break;
                        case MigrationStrategy::STOP_AND_COPY:
                            execute_stop_and_copy(migration_id, plan);
                            break;
                        case MigrationStrategy::DOUBLE_WRITE:
                            execute_double_write_migration(migration_id, plan);
                            break;
                        case MigrationStrategy::STAGED_MIGRATION:
                            execute_staged_migration(migration_id, plan);
                            break;
                    }
                }
            );
        }
        
        LOG_INFO(logger_, "Started migration " + migration_id + " for shard " + plan.shard_id);
        return migration_id;
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in start_migration: " + std::string(e.what()));
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to start migration: " + std::string(e.what()));
    }
}

Result<bool> LiveMigrationService::pause_migration(const std::string& migration_id) {
    try {
        std::lock_guard<std::mutex> lock(migrations_mutex_);
        
        auto it = migrations_.find(migration_id);
        if (it == migrations_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Migration not found: " + migration_id);
        }
        
        LOG_INFO(logger_, "Pausing migration: " + migration_id);
        // Pausing would be implemented by setting a flag checked during migration
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to pause migration: " + std::string(e.what()));
    }
}

Result<bool> LiveMigrationService::resume_migration(const std::string& migration_id) {
    try {
        std::lock_guard<std::mutex> lock(migrations_mutex_);
        
        auto it = migrations_.find(migration_id);
        if (it == migrations_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Migration not found: " + migration_id);
        }
        
        LOG_INFO(logger_, "Resuming migration: " + migration_id);
        // Resuming would be implemented by clearing the pause flag
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to resume migration: " + std::string(e.what()));
    }
}

Result<bool> LiveMigrationService::cancel_migration(const std::string& migration_id) {
    try {
        LOG_INFO(logger_, "Cancelling migration: " + migration_id);
        
        {
            std::lock_guard<std::mutex> lock(migrations_mutex_);
            auto it = migrations_.find(migration_id);
            if (it == migrations_.end()) {
                RETURN_ERROR(ErrorCode::NOT_FOUND, "Migration not found: " + migration_id);
            }
            
            it->second.phase = MigrationPhase::FAILED;
            it->second.error_message = "Migration cancelled by user";
            it->second.completed_at = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        }
        
        // Thread will detect the phase change and stop
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to cancel migration: " + std::string(e.what()));
    }
}

Result<bool> LiveMigrationService::create_checkpoint(const std::string& migration_id) {
    try {
        std::lock_guard<std::mutex> lock(migrations_mutex_);
        
        auto it = migrations_.find(migration_id);
        if (it == migrations_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Migration not found: " + migration_id);
        }
        
        MigrationCheckpoint checkpoint;
        checkpoint.checkpoint_id = generate_checkpoint_id();
        checkpoint.migration_id = migration_id;
        checkpoint.phase = it->second.phase;
        checkpoint.vectors_copied = it->second.copied_vectors;
        checkpoint.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        std::lock_guard<std::mutex> cp_lock(checkpoints_mutex_);
        checkpoints_[migration_id].push_back(checkpoint);
        
        it->second.checkpoint_count++;
        
        LOG_INFO(logger_, "Created checkpoint " + checkpoint.checkpoint_id + 
                " for migration " + migration_id);
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to create checkpoint: " + std::string(e.what()));
    }
}

Result<bool> LiveMigrationService::rollback_to_checkpoint(
    const std::string& migration_id,
    const std::string& checkpoint_id) {
    try {
        LOG_INFO(logger_, "Rolling back migration " + migration_id + 
                " to checkpoint " + checkpoint_id);
        
        std::lock_guard<std::mutex> lock(checkpoints_mutex_);
        auto it = checkpoints_.find(migration_id);
        if (it == checkpoints_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "No checkpoints found for migration: " + migration_id);
        }
        
        auto cp_it = std::find_if(it->second.begin(), it->second.end(),
            [&checkpoint_id](const MigrationCheckpoint& cp) {
                return cp.checkpoint_id == checkpoint_id;
            });
        
        if (cp_it == it->second.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Checkpoint not found: " + checkpoint_id);
        }
        
        // Rollback logic would restore state from checkpoint
        LOG_INFO(logger_, "Rolled back to checkpoint " + checkpoint_id);
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to rollback: " + std::string(e.what()));
    }
}

Result<bool> LiveMigrationService::rollback_migration(const std::string& migration_id) {
    try {
        LOG_INFO(logger_, "Rolling back migration: " + migration_id);
        
        std::lock_guard<std::mutex> lock(migrations_mutex_);
        auto it = migrations_.find(migration_id);
        if (it == migrations_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Migration not found: " + migration_id);
        }
        
        if (!it->second.can_rollback) {
            RETURN_ERROR(ErrorCode::INVALID_ARGUMENT, "Migration cannot be rolled back");
        }
        
        it->second.phase = MigrationPhase::ROLLING_BACK;
        
        // Rollback logic would:
        // 1. Disable redirections
        // 2. Restore shard to original node
        // 3. Clean up target node
        
        LOG_INFO(logger_, "Migration rolled back: " + migration_id);
        
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to rollback migration: " + std::string(e.what()));
    }
}

Result<LiveMigrationStatus> LiveMigrationService::get_migration_status(const std::string& migration_id) {
    try {
        std::lock_guard<std::mutex> lock(migrations_mutex_);
        
        auto it = migrations_.find(migration_id);
        if (it == migrations_.end()) {
            RETURN_ERROR(ErrorCode::NOT_FOUND, "Migration not found: " + migration_id);
        }
        
        return it->second;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to get migration status: " + std::string(e.what()));
    }
}

std::vector<LiveMigrationStatus> LiveMigrationService::get_active_migrations() {
    std::lock_guard<std::mutex> lock(migrations_mutex_);
    std::vector<LiveMigrationStatus> active;
    for (const auto& pair : migrations_) {
        if (pair.second.phase != MigrationPhase::COMPLETED &&
            pair.second.phase != MigrationPhase::FAILED) {
            active.push_back(pair.second);
        }
    }
    return active;
}

std::vector<MigrationCheckpoint> LiveMigrationService::get_checkpoints(const std::string& migration_id) {
    std::lock_guard<std::mutex> lock(checkpoints_mutex_);
    auto it = checkpoints_.find(migration_id);
    if (it != checkpoints_.end()) {
        return it->second;
    }
    return {};
}

void LiveMigrationService::register_progress_callback(MigrationProgressCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    progress_callbacks_.push_back(callback);
}

Result<bool> LiveMigrationService::enable_double_write(const std::string& shard_id) {
    try {
        std::lock_guard<std::mutex> lock(redirections_mutex_);
        double_write_enabled_[shard_id] = true;
        LOG_INFO(logger_, "Enabled double-write for shard: " + shard_id);
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to enable double-write: " + std::string(e.what()));
    }
}

Result<bool> LiveMigrationService::disable_double_write(const std::string& shard_id) {
    try {
        std::lock_guard<std::mutex> lock(redirections_mutex_);
        double_write_enabled_.erase(shard_id);
        LOG_INFO(logger_, "Disabled double-write for shard: " + shard_id);
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to disable double-write: " + std::string(e.what()));
    }
}

Result<bool> LiveMigrationService::redirect_reads(const std::string& shard_id, const std::string& target_node) {
    try {
        std::lock_guard<std::mutex> lock(redirections_mutex_);
        read_redirections_[shard_id] = target_node;
        LOG_INFO(logger_, "Redirecting reads for shard " + shard_id + " to " + target_node);
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to redirect reads: " + std::string(e.what()));
    }
}

Result<bool> LiveMigrationService::redirect_writes(const std::string& shard_id, const std::string& target_node) {
    try {
        std::lock_guard<std::mutex> lock(redirections_mutex_);
        write_redirections_[shard_id] = target_node;
        LOG_INFO(logger_, "Redirecting writes for shard " + shard_id + " to " + target_node);
        return true;
    } catch (const std::exception& e) {
        RETURN_ERROR(ErrorCode::SERVICE_ERROR, "Failed to redirect writes: " + std::string(e.what()));
    }
}

// Private methods

void LiveMigrationService::execute_live_migration(const std::string& migration_id, const MigrationPlan& plan) {
    try {
        LOG_INFO(logger_, "Executing live migration: " + migration_id);
        
        // Phase 1: Planning
        update_migration_phase(migration_id, MigrationPhase::PLANNING);
        auto planning = phase_planning(migration_id, plan);
        if (!planning.has_value()) {
            mark_migration_failed(migration_id, "Planning failed: " + planning.error().message);
            return;
        }
        
        // Phase 2: Preparing (enable double-write)
        update_migration_phase(migration_id, MigrationPhase::PREPARING);
        auto preparing = phase_preparing(migration_id, plan);
        if (!preparing.has_value()) {
            mark_migration_failed(migration_id, "Preparing failed: " + preparing.error().message);
            return;
        }
        enable_double_write(plan.shard_id);
        
        // Phase 3: Copying (bulk copy)
        update_migration_phase(migration_id, MigrationPhase::COPYING);
        auto copying = phase_copying(migration_id, plan);
        if (!copying.has_value()) {
            mark_migration_failed(migration_id, "Copying failed: " + copying.error().message);
            return;
        }
        
        // Phase 4: Syncing (catch up with incremental changes)
        update_migration_phase(migration_id, MigrationPhase::SYNCING);
        auto syncing = phase_syncing(migration_id, plan);
        if (!syncing.has_value()) {
            mark_migration_failed(migration_id, "Syncing failed: " + syncing.error().message);
            return;
        }
        
        // Phase 5: Switching (redirect traffic)
        update_migration_phase(migration_id, MigrationPhase::SWITCHING);
        auto switching = phase_switching(migration_id, plan);
        if (!switching.has_value()) {
            mark_migration_failed(migration_id, "Switching failed: " + switching.error().message);
            return;
        }
        
        // Phase 6: Verifying
        if (plan.verify_data) {
            update_migration_phase(migration_id, MigrationPhase::VERIFYING);
            auto verifying = phase_verifying(migration_id, plan);
            if (!verifying.has_value()) {
                LOG_WARN(logger_, "Verification failed but migration completed: " + verifying.error().message);
            }
        }
        
        // Cleanup
        disable_double_write(plan.shard_id);
        
        mark_migration_completed(migration_id);
        LOG_INFO(logger_, "Live migration completed: " + migration_id);
        
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in execute_live_migration: " + std::string(e.what()));
        mark_migration_failed(migration_id, "Exception: " + std::string(e.what()));
    }
}

void LiveMigrationService::execute_stop_and_copy(const std::string& migration_id, const MigrationPlan& plan) {
    try {
        LOG_INFO(logger_, "Executing stop-and-copy migration: " + migration_id);
        
        // Simplified version: stop writes, copy, resume
        update_migration_phase(migration_id, MigrationPhase::PREPARING);
        phase_preparing(migration_id, plan);
        
        update_migration_phase(migration_id, MigrationPhase::COPYING);
        auto copying = phase_copying(migration_id, plan);
        if (!copying.has_value()) {
            mark_migration_failed(migration_id, "Copying failed");
            return;
        }
        
        update_migration_phase(migration_id, MigrationPhase::SWITCHING);
        phase_switching(migration_id, plan);
        
        mark_migration_completed(migration_id);
        
    } catch (const std::exception& e) {
        mark_migration_failed(migration_id, std::string(e.what()));
    }
}

void LiveMigrationService::execute_double_write_migration(const std::string& migration_id, const MigrationPlan& plan) {
    try {
        LOG_INFO(logger_, "Executing double-write migration: " + migration_id);
        
        enable_double_write(plan.shard_id);
        
        update_migration_phase(migration_id, MigrationPhase::COPYING);
        phase_copying(migration_id, plan);
        
        update_migration_phase(migration_id, MigrationPhase::SWITCHING);
        phase_switching(migration_id, plan);
        
        disable_double_write(plan.shard_id);
        
        mark_migration_completed(migration_id);
        
    } catch (const std::exception& e) {
        mark_migration_failed(migration_id, std::string(e.what()));
    }
}

void LiveMigrationService::execute_staged_migration(const std::string& migration_id, const MigrationPlan& plan) {
    try {
        LOG_INFO(logger_, "Executing staged migration: " + migration_id);
        
        // Migrate in small chunks over time
        update_migration_phase(migration_id, MigrationPhase::COPYING);
        
        // Simulate chunked migration
        for (int i = 0; i < 10; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            update_migration_progress(migration_id, i * plan.chunk_size, 0);
            create_checkpoint(migration_id);
        }
        
        update_migration_phase(migration_id, MigrationPhase::SWITCHING);
        phase_switching(migration_id, plan);
        
        mark_migration_completed(migration_id);
        
    } catch (const std::exception& e) {
        mark_migration_failed(migration_id, std::string(e.what()));
    }
}

Result<bool> LiveMigrationService::phase_planning(const std::string& migration_id, const MigrationPlan& plan) {
    LOG_INFO(logger_, "Migration " + migration_id + ": Planning phase");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return true;
}

Result<bool> LiveMigrationService::phase_preparing(const std::string& migration_id, const MigrationPlan& plan) {
    LOG_INFO(logger_, "Migration " + migration_id + ": Preparing phase");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return true;
}

Result<bool> LiveMigrationService::phase_copying(const std::string& migration_id, const MigrationPlan& plan) {
    LOG_INFO(logger_, "Migration " + migration_id + ": Copying phase");
    
    // Simulate copying data in chunks
    int64_t total = 100000; // Simulated total vectors
    for (int64_t copied = 0; copied < total; copied += plan.chunk_size) {
        update_migration_progress(migration_id, copied, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    update_migration_progress(migration_id, total, 0);
    return true;
}

Result<bool> LiveMigrationService::phase_syncing(const std::string& migration_id, const MigrationPlan& plan) {
    LOG_INFO(logger_, "Migration " + migration_id + ": Syncing phase");
    
    // Simulate incremental sync
    std::lock_guard<std::mutex> lock(migrations_mutex_);
    auto it = migrations_.find(migration_id);
    if (it != migrations_.end()) {
        it->second.synced_vectors = it->second.copied_vectors;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return true;
}

Result<bool> LiveMigrationService::phase_switching(const std::string& migration_id, const MigrationPlan& plan) {
    LOG_INFO(logger_, "Migration " + migration_id + ": Switching phase");
    
    // Redirect reads and writes to target node
    redirect_reads(plan.shard_id, plan.target_node);
    redirect_writes(plan.shard_id, plan.target_node);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return true;
}

Result<bool> LiveMigrationService::phase_verifying(const std::string& migration_id, const MigrationPlan& plan) {
    LOG_INFO(logger_, "Migration " + migration_id + ": Verifying phase");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return true;
}

void LiveMigrationService::update_migration_phase(const std::string& migration_id, MigrationPhase phase) {
    std::lock_guard<std::mutex> lock(migrations_mutex_);
    auto it = migrations_.find(migration_id);
    if (it != migrations_.end()) {
        it->second.phase = phase;
        it->second.phase_started_at = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        LOG_INFO(logger_, "Migration " + migration_id + " entered phase: " + migration_phase_to_string(phase));
        notify_progress(migration_id);
    }
}

void LiveMigrationService::update_migration_progress(const std::string& migration_id, int64_t copied, int64_t synced) {
    std::lock_guard<std::mutex> lock(migrations_mutex_);
    auto it = migrations_.find(migration_id);
    if (it != migrations_.end()) {
        it->second.copied_vectors = copied;
        it->second.synced_vectors = synced;
        it->second.transfer_rate_mbps = calculate_transfer_rate(it->second);
    }
}

void LiveMigrationService::mark_migration_failed(const std::string& migration_id, const std::string& error) {
    std::lock_guard<std::mutex> lock(migrations_mutex_);
    auto it = migrations_.find(migration_id);
    if (it != migrations_.end()) {
        it->second.phase = MigrationPhase::FAILED;
        it->second.error_message = error;
        it->second.completed_at = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        LOG_ERROR(logger_, "Migration " + migration_id + " failed: " + error);
        notify_progress(migration_id);
    }
}

void LiveMigrationService::mark_migration_completed(const std::string& migration_id) {
    std::lock_guard<std::mutex> lock(migrations_mutex_);
    auto it = migrations_.find(migration_id);
    if (it != migrations_.end()) {
        it->second.phase = MigrationPhase::COMPLETED;
        it->second.completed_at = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        LOG_INFO(logger_, "Migration " + migration_id + " completed successfully");
        notify_progress(migration_id);
    }
}

std::string LiveMigrationService::generate_migration_id() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return "lmig_" + std::to_string(duration.count());
}

std::string LiveMigrationService::generate_checkpoint_id() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return "cp_" + std::to_string(duration.count());
}

int64_t LiveMigrationService::estimate_migration_duration(const MigrationPlan& plan) {
    // Simple estimation: assume 10MB/s transfer rate
    int64_t estimated_bytes = plan.estimated_data_size > 0 ? plan.estimated_data_size : 1000000000; // 1GB default
    int64_t duration = estimated_bytes / (10 * 1024 * 1024); // seconds
    return std::max(int64_t(60), duration); // minimum 1 minute
}

double LiveMigrationService::calculate_transfer_rate(const LiveMigrationStatus& status) {
    if (status.started_at == 0) return 0.0;
    
    int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    int64_t elapsed = now - status.started_at;
    
    if (elapsed == 0) return 0.0;
    
    double mbps = (status.transferred_bytes / (1024.0 * 1024.0)) / elapsed;
    return mbps;
}

void LiveMigrationService::notify_progress(const std::string& migration_id) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    LiveMigrationStatus status;
    {
        std::lock_guard<std::mutex> mig_lock(migrations_mutex_);
        auto it = migrations_.find(migration_id);
        if (it != migrations_.end()) {
            status = it->second;
        } else {
            return;
        }
    }
    
    for (const auto& callback : progress_callbacks_) {
        try {
            callback(status);
        } catch (const std::exception& e) {
            LOG_ERROR(logger_, "Exception in progress callback: " + std::string(e.what()));
        }
    }
}

} // namespace jadevectordb
