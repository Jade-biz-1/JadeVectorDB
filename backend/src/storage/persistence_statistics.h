#pragma once

#include <string>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <chrono>
#include <cstdint>

namespace jadevectordb {

/**
 * @brief Per-database statistics (internal, with atomic counters)
 */
struct DatabaseStats {
    std::atomic<uint64_t> read_count{0};
    std::atomic<uint64_t> write_count{0};
    std::atomic<uint64_t> delete_count{0};
    std::atomic<uint64_t> update_count{0};
    std::atomic<uint64_t> compaction_count{0};
    std::atomic<uint64_t> index_resize_count{0};
    std::atomic<uint64_t> snapshot_count{0};
    std::atomic<uint64_t> wal_checkpoint_count{0};
    
    std::atomic<uint64_t> bytes_read{0};
    std::atomic<uint64_t> bytes_written{0};
    std::atomic<uint64_t> bytes_compacted{0};
    
    std::atomic<uint64_t> total_read_time_us{0};
    std::atomic<uint64_t> total_write_time_us{0};
    std::atomic<uint64_t> total_compaction_time_us{0};
    
    std::atomic<uint64_t> last_read_timestamp{0};
    std::atomic<uint64_t> last_write_timestamp{0};
    std::atomic<uint64_t> last_compaction_timestamp{0};
    
    // Reset statistics
    void reset() {
        read_count = 0;
        write_count = 0;
        delete_count = 0;
        update_count = 0;
        compaction_count = 0;
        index_resize_count = 0;
        snapshot_count = 0;
        wal_checkpoint_count = 0;
        bytes_read = 0;
        bytes_written = 0;
        bytes_compacted = 0;
        total_read_time_us = 0;
        total_write_time_us = 0;
        total_compaction_time_us = 0;
        last_read_timestamp = 0;
        last_write_timestamp = 0;
        last_compaction_timestamp = 0;
    }
};

/**
 * @brief Snapshot of database statistics (copyable, for reading)
 */
struct DatabaseStatsSnapshot {
    uint64_t read_count = 0;
    uint64_t write_count = 0;
    uint64_t delete_count = 0;
    uint64_t update_count = 0;
    uint64_t compaction_count = 0;
    uint64_t index_resize_count = 0;
    uint64_t snapshot_count = 0;
    uint64_t wal_checkpoint_count = 0;
    
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
    uint64_t bytes_compacted = 0;
    
    uint64_t total_read_time_us = 0;
    uint64_t total_write_time_us = 0;
    uint64_t total_compaction_time_us = 0;
    
    uint64_t last_read_timestamp = 0;
    uint64_t last_write_timestamp = 0;
    uint64_t last_compaction_timestamp = 0;
};

/**
 * @brief Aggregated system-wide statistics
 */
struct SystemStats {
    uint64_t total_databases = 0;
    uint64_t total_read_count = 0;
    uint64_t total_write_count = 0;
    uint64_t total_delete_count = 0;
    uint64_t total_update_count = 0;
    uint64_t total_compaction_count = 0;
    uint64_t total_index_resize_count = 0;
    uint64_t total_snapshot_count = 0;
    uint64_t total_wal_checkpoint_count = 0;
    uint64_t total_bytes_read = 0;
    uint64_t total_bytes_written = 0;
    uint64_t total_bytes_compacted = 0;
    double avg_read_latency_ms = 0.0;
    double avg_write_latency_ms = 0.0;
    double avg_compaction_time_ms = 0.0;
    uint64_t uptime_seconds = 0;
};

/**
 * @brief Operation timer for automatic timing
 */
class OperationTimer {
public:
    explicit OperationTimer(std::atomic<uint64_t>& total_time, std::atomic<uint64_t>& timestamp)
        : total_time_(total_time)
        , timestamp_(timestamp)
        , start_(std::chrono::steady_clock::now()) {}
    
    ~OperationTimer() {
        auto end = std::chrono::steady_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        total_time_.fetch_add(duration_us);
        timestamp_.store(std::chrono::system_clock::now().time_since_epoch().count());
    }

private:
    std::atomic<uint64_t>& total_time_;
    std::atomic<uint64_t>& timestamp_;
    std::chrono::steady_clock::time_point start_;
};

/**
 * @brief Persistence statistics tracker
 * 
 * Tracks all persistence operations:
 * - Read/write counts and bytes
 * - Compaction statistics
 * - Index resize events
 * - Snapshot operations
 * - WAL checkpoints
 * - Operation latencies
 */
class PersistenceStatistics {
public:
    /**
     * @brief Get singleton instance
     */
    static PersistenceStatistics& instance();

    // Disable copy and move
    PersistenceStatistics(const PersistenceStatistics&) = delete;
    PersistenceStatistics& operator=(const PersistenceStatistics&) = delete;

    /**
     * @brief Record a read operation
     * @param database_id Database identifier
     * @param bytes_read Number of bytes read
     * @return Timer object (destroyed when operation completes)
     */
    OperationTimer record_read(const std::string& database_id, uint64_t bytes_read);

    /**
     * @brief Record a write operation
     * @param database_id Database identifier
     * @param bytes_written Number of bytes written
     * @return Timer object
     */
    OperationTimer record_write(const std::string& database_id, uint64_t bytes_written);

    /**
     * @brief Record a delete operation
     * @param database_id Database identifier
     */
    void record_delete(const std::string& database_id);

    /**
     * @brief Record an update operation
     * @param database_id Database identifier
     * @param bytes_written Number of bytes written
     */
    void record_update(const std::string& database_id, uint64_t bytes_written);

    /**
     * @brief Record a compaction operation
     * @param database_id Database identifier
     * @param bytes_compacted Number of bytes reclaimed
     * @return Timer object
     */
    OperationTimer record_compaction(const std::string& database_id, uint64_t bytes_compacted);

    /**
     * @brief Record an index resize operation
     * @param database_id Database identifier
     */
    void record_index_resize(const std::string& database_id);

    /**
     * @brief Record a snapshot operation
     * @param database_id Database identifier
     */
    void record_snapshot(const std::string& database_id);

    /**
     * @brief Record a WAL checkpoint operation
     * @param database_id Database identifier
     */
    void record_wal_checkpoint(const std::string& database_id);

    /**
     * @brief Get statistics for a specific database
     * @param database_id Database identifier
     * @return Database statistics snapshot
     */
    DatabaseStatsSnapshot get_database_stats(const std::string& database_id);

    /**
     * @brief Get system-wide aggregated statistics
     * @return System statistics
     */
    SystemStats get_system_stats();

    /**
     * @brief Reset statistics for a database
     * @param database_id Database identifier
     */
    void reset_database_stats(const std::string& database_id);

    /**
     * @brief Reset all statistics
     */
    void reset_all_stats();

    /**
     * @brief Get list of all databases being tracked
     * @return Vector of database IDs
     */
    std::vector<std::string> get_tracked_databases();

private:
    PersistenceStatistics();
    ~PersistenceStatistics() = default;

    std::unordered_map<std::string, DatabaseStats> database_stats_;
    std::mutex stats_mutex_;
    std::chrono::steady_clock::time_point start_time_;
    
    DatabaseStats& get_or_create_stats(const std::string& database_id);
};

} // namespace jadevectordb
