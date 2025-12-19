#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <optional>

namespace jadevectordb {

// Forward declaration
class MemoryMappedVectorStore;

/**
 * @brief Snapshot metadata
 */
struct SnapshotMetadata {
    std::string snapshot_id;
    std::string database_id;
    uint64_t timestamp;
    uint64_t vector_count;
    uint64_t file_size;
    std::string checksum;  // SHA-256 checksum
    std::string description;
};

/**
 * @brief Snapshot statistics
 */
struct SnapshotStats {
    uint64_t vectors_snapshotted = 0;
    uint64_t bytes_written = 0;
    double duration_seconds = 0.0;
    bool success = false;
    std::string error_message;
    SnapshotMetadata metadata;
};

/**
 * @brief Restore statistics
 */
struct SnapshotRestoreStats {
    uint64_t vectors_restored = 0;
    uint64_t bytes_read = 0;
    double duration_seconds = 0.0;
    bool success = false;
    std::string error_message;
};

/**
 * @brief Database snapshot manager
 * 
 * Provides point-in-time snapshots:
 * - Create immutable snapshots of database state
 * - Restore from snapshots
 * - List and manage snapshots
 * - Verify snapshot integrity
 */
class SnapshotManager {
public:
    /**
     * @brief Constructor
     * @param store Reference to vector store
     * @param snapshot_directory Directory for storing snapshots
     */
    explicit SnapshotManager(MemoryMappedVectorStore& store,
                            const std::string& snapshot_directory);
    
    /**
     * @brief Destructor
     */
    ~SnapshotManager();

    // Disable copy and move
    SnapshotManager(const SnapshotManager&) = delete;
    SnapshotManager& operator=(const SnapshotManager&) = delete;

    /**
     * @brief Create snapshot of database
     * @param database_id Database identifier
     * @param description Optional description
     * @return Snapshot statistics
     */
    SnapshotStats create_snapshot(const std::string& database_id,
                                 const std::string& description = "");

    /**
     * @brief Restore database from snapshot
     * @param snapshot_id Snapshot identifier
     * @param target_database_id Target database ID (can differ from original)
     * @return Restore statistics
     */
    SnapshotRestoreStats restore_from_snapshot(const std::string& snapshot_id,
                                               const std::string& target_database_id);

    /**
     * @brief List all snapshots for a database
     * @param database_id Database identifier (empty for all databases)
     * @return Vector of snapshot metadata
     */
    std::vector<SnapshotMetadata> list_snapshots(const std::string& database_id = "");

    /**
     * @brief Get snapshot metadata
     * @param snapshot_id Snapshot identifier
     * @return Snapshot metadata, or nullopt if not found
     */
    std::optional<SnapshotMetadata> get_snapshot_metadata(const std::string& snapshot_id);

    /**
     * @brief Delete a snapshot
     * @param snapshot_id Snapshot identifier
     * @return true if deleted successfully
     */
    bool delete_snapshot(const std::string& snapshot_id);

    /**
     * @brief Verify snapshot integrity
     * @param snapshot_id Snapshot identifier
     * @return true if snapshot is valid
     */
    bool verify_snapshot(const std::string& snapshot_id);

    /**
     * @brief Cleanup old snapshots (keep N most recent)
     * @param database_id Database identifier
     * @param keep_count Number of snapshots to keep
     * @return Number of snapshots deleted
     */
    int cleanup_old_snapshots(const std::string& database_id, int keep_count);

    /**
     * @brief Get total size of all snapshots
     * @return Total size in bytes
     */
    uint64_t get_total_snapshot_size();

private:
    MemoryMappedVectorStore& store_;
    std::string snapshot_directory_;
    
    std::string generate_snapshot_id();
    std::string get_snapshot_data_path(const std::string& snapshot_id) const;
    std::string get_snapshot_meta_path(const std::string& snapshot_id) const;
    std::string calculate_file_checksum(const std::string& file_path) const;
    bool write_metadata(const SnapshotMetadata& metadata);
    std::optional<SnapshotMetadata> read_metadata(const std::string& meta_path);
};

} // namespace jadevectordb
