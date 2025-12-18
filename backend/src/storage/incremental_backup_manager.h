#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <optional>

namespace jadevectordb {

// Forward declaration
class MemoryMappedVectorStore;

/**
 * @brief Backup metadata
 */
struct BackupMetadata {
    std::string backup_id;
    std::string database_id;
    uint64_t timestamp;
    bool is_full_backup;
    std::string parent_backup_id;  // For incremental backups
    uint64_t vector_count;
    uint64_t size_bytes;
    std::string checksum;  // SHA-256 checksum
};

/**
 * @brief Backup statistics
 */
struct BackupStats {
    uint64_t vectors_backed_up = 0;
    uint64_t bytes_written = 0;
    double duration_seconds = 0.0;
    bool success = false;
    std::string error_message;
    BackupMetadata metadata;
};

/**
 * @brief Restore statistics
 */
struct RestoreStats {
    uint64_t vectors_restored = 0;
    uint64_t bytes_read = 0;
    double duration_seconds = 0.0;
    bool success = false;
    std::string error_message;
};

/**
 * @brief Incremental backup manager
 * 
 * Provides delta-based backups:
 * - Tracks modified vectors since last backup
 * - Creates full and incremental backups
 * - Restores from backup chain
 * - Verifies backup integrity with checksums
 */
class IncrementalBackupManager {
public:
    /**
     * @brief Constructor
     * @param store Reference to vector store
     * @param backup_directory Directory for storing backups
     */
    explicit IncrementalBackupManager(MemoryMappedVectorStore& store,
                                     const std::string& backup_directory);
    
    /**
     * @brief Destructor
     */
    ~IncrementalBackupManager();

    // Disable copy and move
    IncrementalBackupManager(const IncrementalBackupManager&) = delete;
    IncrementalBackupManager& operator=(const IncrementalBackupManager&) = delete;

    /**
     * @brief Create full backup of database
     * @param database_id Database identifier
     * @return Backup statistics
     */
    BackupStats create_full_backup(const std::string& database_id);

    /**
     * @brief Create incremental backup (delta since last backup)
     * @param database_id Database identifier
     * @return Backup statistics
     */
    BackupStats create_incremental_backup(const std::string& database_id);

    /**
     * @brief Restore database from backup
     * @param backup_id Backup identifier
     * @param target_database_id Target database ID (can differ from original)
     * @return Restore statistics
     */
    RestoreStats restore_from_backup(const std::string& backup_id,
                                    const std::string& target_database_id);

    /**
     * @brief List available backups for a database
     * @param database_id Database identifier (empty = all databases)
     * @return Vector of backup metadata
     */
    std::vector<BackupMetadata> list_backups(const std::string& database_id = "") const;

    /**
     * @brief Delete a backup
     * @param backup_id Backup identifier
     * @return true if deleted successfully
     */
    bool delete_backup(const std::string& backup_id);

    /**
     * @brief Verify backup integrity
     * @param backup_id Backup identifier
     * @return true if backup is valid
     */
    bool verify_backup(const std::string& backup_id) const;

    /**
     * @brief Get backup chain for a backup (full + all incrementals)
     * @param backup_id Backup identifier
     * @return Vector of backup IDs in restore order
     */
    std::vector<std::string> get_backup_chain(const std::string& backup_id) const;

    /**
     * @brief Enable change tracking for a database
     * @param database_id Database identifier
     */
    void enable_change_tracking(const std::string& database_id);

    /**
     * @brief Disable change tracking for a database
     * @param database_id Database identifier
     */
    void disable_change_tracking(const std::string& database_id);

    /**
     * @brief Check if change tracking is enabled
     * @param database_id Database identifier
     */
    bool is_change_tracking_enabled(const std::string& database_id) const;

    /**
     * @brief Record vector modification (called by store)
     * @param database_id Database identifier
     * @param vector_id Vector identifier
     */
    void record_vector_change(const std::string& database_id, const std::string& vector_id);

    /**
     * @brief Clear change tracking for a database
     * @param database_id Database identifier
     */
    void clear_change_tracking(const std::string& database_id);

private:
    // Vector store reference
    MemoryMappedVectorStore& store_;
    
    // Backup directory
    std::string backup_directory_;
    
    // Change tracking (database_id -> set of modified vector_ids)
    std::unordered_map<std::string, std::unordered_set<std::string>> changed_vectors_;
    std::unordered_map<std::string, bool> tracking_enabled_;
    std::mutex tracking_mutex_;
    
    // Last backup timestamp per database
    std::unordered_map<std::string, uint64_t> last_backup_time_;
    std::unordered_map<std::string, std::string> last_backup_id_;
    std::mutex backup_state_mutex_;
    
    // Backup ID counter for uniqueness
    mutable std::atomic<uint64_t> backup_counter_{0};
    
    // Helper methods
    std::string generate_backup_id() const;
    std::string get_backup_path(const std::string& backup_id) const;
    std::string get_metadata_path(const std::string& backup_id) const;
    
    bool write_backup_file(const std::string& backup_path,
                          const std::string& database_id,
                          const std::vector<std::string>& vector_ids);
    
    bool read_backup_file(const std::string& backup_path,
                         std::vector<std::pair<std::string, std::vector<float>>>& vectors) const;
    
    bool write_metadata(const BackupMetadata& metadata) const;
    std::optional<BackupMetadata> read_metadata(const std::string& backup_id) const;
    
    std::string calculate_checksum(const std::string& file_path) const;
    uint64_t get_current_timestamp() const;
};

} // namespace jadevectordb
