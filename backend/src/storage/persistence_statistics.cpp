#include "persistence_statistics.h"
#include <algorithm>

namespace jadevectordb {

PersistenceStatistics::PersistenceStatistics()
    : start_time_(std::chrono::steady_clock::now()) {
}

PersistenceStatistics& PersistenceStatistics::instance() {
    static PersistenceStatistics instance;
    return instance;
}

DatabaseStats& PersistenceStatistics::get_or_create_stats(const std::string& database_id) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return database_stats_[database_id];
}

OperationTimer PersistenceStatistics::record_read(const std::string& database_id, uint64_t bytes_read) {
    auto& stats = get_or_create_stats(database_id);
    stats.read_count.fetch_add(1);
    stats.bytes_read.fetch_add(bytes_read);
    return OperationTimer(stats.total_read_time_us, stats.last_read_timestamp);
}

OperationTimer PersistenceStatistics::record_write(const std::string& database_id, uint64_t bytes_written) {
    auto& stats = get_or_create_stats(database_id);
    stats.write_count.fetch_add(1);
    stats.bytes_written.fetch_add(bytes_written);
    return OperationTimer(stats.total_write_time_us, stats.last_write_timestamp);
}

void PersistenceStatistics::record_delete(const std::string& database_id) {
    auto& stats = get_or_create_stats(database_id);
    stats.delete_count.fetch_add(1);
}

void PersistenceStatistics::record_update(const std::string& database_id, uint64_t bytes_written) {
    auto& stats = get_or_create_stats(database_id);
    stats.update_count.fetch_add(1);
    stats.bytes_written.fetch_add(bytes_written);
}

OperationTimer PersistenceStatistics::record_compaction(const std::string& database_id, uint64_t bytes_compacted) {
    auto& stats = get_or_create_stats(database_id);
    stats.compaction_count.fetch_add(1);
    stats.bytes_compacted.fetch_add(bytes_compacted);
    return OperationTimer(stats.total_compaction_time_us, stats.last_compaction_timestamp);
}

void PersistenceStatistics::record_index_resize(const std::string& database_id) {
    auto& stats = get_or_create_stats(database_id);
    stats.index_resize_count.fetch_add(1);
}

void PersistenceStatistics::record_snapshot(const std::string& database_id) {
    auto& stats = get_or_create_stats(database_id);
    stats.snapshot_count.fetch_add(1);
}

void PersistenceStatistics::record_wal_checkpoint(const std::string& database_id) {
    auto& stats = get_or_create_stats(database_id);
    stats.wal_checkpoint_count.fetch_add(1);
}

DatabaseStatsSnapshot PersistenceStatistics::get_database_stats(const std::string& database_id) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    auto it = database_stats_.find(database_id);
    if (it != database_stats_.end()) {
        // Create snapshot by loading atomic values
        DatabaseStatsSnapshot snapshot;
        snapshot.read_count = it->second.read_count.load();
        snapshot.write_count = it->second.write_count.load();
        snapshot.delete_count = it->second.delete_count.load();
        snapshot.update_count = it->second.update_count.load();
        snapshot.compaction_count = it->second.compaction_count.load();
        snapshot.index_resize_count = it->second.index_resize_count.load();
        snapshot.snapshot_count = it->second.snapshot_count.load();
        snapshot.wal_checkpoint_count = it->second.wal_checkpoint_count.load();
        snapshot.bytes_read = it->second.bytes_read.load();
        snapshot.bytes_written = it->second.bytes_written.load();
        snapshot.bytes_compacted = it->second.bytes_compacted.load();
        snapshot.total_read_time_us = it->second.total_read_time_us.load();
        snapshot.total_write_time_us = it->second.total_write_time_us.load();
        snapshot.total_compaction_time_us = it->second.total_compaction_time_us.load();
        snapshot.last_read_timestamp = it->second.last_read_timestamp.load();
        snapshot.last_write_timestamp = it->second.last_write_timestamp.load();
        snapshot.last_compaction_timestamp = it->second.last_compaction_timestamp.load();
        return snapshot;
    }
    
    return DatabaseStatsSnapshot{};
}

SystemStats PersistenceStatistics::get_system_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    SystemStats system;
    system.total_databases = database_stats_.size();
    
    uint64_t total_read_time_us = 0;
    uint64_t total_write_time_us = 0;
    uint64_t total_compaction_time_us = 0;
    
    for (const auto& [db_id, stats] : database_stats_) {
        system.total_read_count += stats.read_count.load();
        system.total_write_count += stats.write_count.load();
        system.total_delete_count += stats.delete_count.load();
        system.total_update_count += stats.update_count.load();
        system.total_compaction_count += stats.compaction_count.load();
        system.total_index_resize_count += stats.index_resize_count.load();
        system.total_snapshot_count += stats.snapshot_count.load();
        system.total_wal_checkpoint_count += stats.wal_checkpoint_count.load();
        system.total_bytes_read += stats.bytes_read.load();
        system.total_bytes_written += stats.bytes_written.load();
        system.total_bytes_compacted += stats.bytes_compacted.load();
        
        total_read_time_us += stats.total_read_time_us.load();
        total_write_time_us += stats.total_write_time_us.load();
        total_compaction_time_us += stats.total_compaction_time_us.load();
    }
    
    // Calculate average latencies
    if (system.total_read_count > 0) {
        system.avg_read_latency_ms = (total_read_time_us / static_cast<double>(system.total_read_count)) / 1000.0;
    }
    
    if (system.total_write_count > 0) {
        system.avg_write_latency_ms = (total_write_time_us / static_cast<double>(system.total_write_count)) / 1000.0;
    }
    
    if (system.total_compaction_count > 0) {
        system.avg_compaction_time_ms = (total_compaction_time_us / static_cast<double>(system.total_compaction_count)) / 1000.0;
    }
    
    // Calculate uptime
    auto now = std::chrono::steady_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
    system.uptime_seconds = uptime.count();
    
    return system;
}

void PersistenceStatistics::reset_database_stats(const std::string& database_id) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    auto it = database_stats_.find(database_id);
    if (it != database_stats_.end()) {
        it->second.reset();
    }
}

void PersistenceStatistics::reset_all_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    for (auto& [db_id, stats] : database_stats_) {
        stats.reset();
    }
    
    start_time_ = std::chrono::steady_clock::now();
}

std::vector<std::string> PersistenceStatistics::get_tracked_databases() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    std::vector<std::string> databases;
    databases.reserve(database_stats_.size());
    
    for (const auto& [db_id, stats] : database_stats_) {
        databases.push_back(db_id);
    }
    
    return databases;
}

} // namespace jadevectordb
