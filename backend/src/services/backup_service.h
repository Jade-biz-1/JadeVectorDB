#ifndef JADEVECTORDB_BACKUP_SERVICE_H
#define JADEVECTORDB_BACKUP_SERVICE_H

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <functional>

#include "lib/error_handling.h"
#include "lib/logging.h"
#include "services/database_layer.h"
#include "models/database.h"
#include "models/vector.h"

namespace jadevectordb {

enum class BackupType {
    FULL,
    INCREMENTAL,
    SNAPSHOT
};

enum class BackupStatus {
    PENDING,
    IN_PROGRESS,
    COMPLETED,
    FAILED,
    CANCELLED
};

enum class RecoveryStatus {
    PENDING,
    IN_PROGRESS,
    COMPLETED,
    FAILED
};

struct BackupConfig {
    std::string backup_name;
    std::string description;
    std::string storage_path;
    std::string compression_method;  // "none", "gzip", "lz4", etc.
    bool encrypt_backup;
    std::string encryption_key;  // In a real system, this would be handled more securely
    int retention_days;
    bool include_indexes;
    bool include_metadata;
    std::vector<std::string> databases_to_backup;  // Empty means all databases
    
    BackupConfig() : 
        compression_method("gzip"), 
        encrypt_backup(false), 
        retention_days(30),
        include_indexes(true),
        include_metadata(true) {}
};

struct RecoveryConfig {
    std::string backup_path;
    std::string target_database;  // Empty means restore to original location
    bool restore_indexes;
    bool restore_metadata;
    bool overwrite_existing;
    std::string encryption_key;  // In a real system, this would be handled more securely
};

struct BackupInfo {
    std::string backup_id;
    std::string backup_name;
    BackupType type;
    BackupStatus status;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    std::string storage_path;
    size_t backup_size_bytes;
    std::vector<std::string> databases_backed_up;
    std::string error_message;
    std::string checksum;
    
    BackupInfo() : type(BackupType::FULL), status(BackupStatus::PENDING), backup_size_bytes(0) {}
};

struct RecoveryInfo {
    std::string recovery_id;
    std::string backup_id;
    RecoveryStatus status;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    std::vector<std::string> databases_restored;
    std::string error_message;
    
    RecoveryInfo() : status(RecoveryStatus::PENDING) {}
};

class BackupService {
private:
    std::shared_ptr<logging::Logger> logger_;
    std::unique_ptr<DatabaseLayer> db_layer_;
    mutable std::mutex backup_mutex_;
    
    // Active backups and recoveries
    std::unordered_map<std::string, BackupInfo> active_backups_;
    std::unordered_map<std::string, RecoveryInfo> active_recoveries_;
    
    // Configuration
    std::string default_backup_path_;
    std::string default_recovery_path_;
    int max_concurrent_backups_;
    std::chrono::milliseconds backup_timeout_;
    
    // Callbacks
    std::function<void(const BackupInfo&)> on_backup_start_callback_;
    std::function<void(const BackupInfo&)> on_backup_complete_callback_;
    std::function<void(const BackupInfo&)> on_backup_error_callback_;
    std::function<void(const RecoveryInfo&)> on_recovery_start_callback_;
    std::function<void(const RecoveryInfo&)> on_recovery_complete_callback_;
    std::function<void(const RecoveryInfo&)> on_recovery_error_callback_;

public:
    explicit BackupService(std::unique_ptr<DatabaseLayer> db_layer = nullptr);
    ~BackupService() = default;
    
    // Initialize the backup service
    Result<void> initialize();
    
    // Configuration methods
    void set_default_backup_path(const std::string& path) { default_backup_path_ = path; }
    void set_default_recovery_path(const std::string& path) { default_recovery_path_ = path; }
    void set_max_concurrent_backups(int max) { max_concurrent_backups_ = max; }
    void set_backup_timeout(std::chrono::milliseconds timeout) { backup_timeout_ = timeout; }
    
    // Backup operations
    Result<std::string> create_backup(const BackupConfig& config);
    Result<BackupInfo> get_backup_info(const std::string& backup_id) const;
    Result<std::vector<BackupInfo>> list_backups() const;
    Result<BackupStatus> get_backup_status(const std::string& backup_id) const;
    Result<bool> cancel_backup(const std::string& backup_id);
    Result<bool> delete_backup(const std::string& backup_id);
    
    // Recovery operations
    Result<std::string> restore_backup(const RecoveryConfig& config);
    Result<RecoveryInfo> get_recovery_info(const std::string& recovery_id) const;
    Result<std::vector<RecoveryInfo>> list_recoveries() const;
    Result<RecoveryStatus> get_recovery_status(const std::string& recovery_id) const;
    
    // Verification operations
    Result<bool> verify_backup_integrity(const std::string& backup_path) const;
    Result<bool> validate_backup_compatibility(const std::string& backup_path) const;
    
    // Set callback functions
    void set_backup_start_callback(std::function<void(const BackupInfo&)> callback) {
        on_backup_start_callback_ = callback;
    }
    void set_backup_complete_callback(std::function<void(const BackupInfo&)> callback) {
        on_backup_complete_callback_ = callback;
    }
    void set_backup_error_callback(std::function<void(const BackupInfo&)> callback) {
        on_backup_error_callback_ = callback;
    }
    void set_recovery_start_callback(std::function<void(const RecoveryInfo&)> callback) {
        on_recovery_start_callback_ = callback;
    }
    void set_recovery_complete_callback(std::function<void(const RecoveryInfo&)> callback) {
        on_recovery_complete_callback_ = callback;
    }
    void set_recovery_error_callback(std::function<void(const RecoveryInfo&)> callback) {
        on_recovery_error_callback_ = callback;
    }
    
    // Utility methods
    std::string backup_type_to_string(BackupType type) const;
    std::string backup_status_to_string(BackupStatus status) const;
    std::string recovery_status_to_string(RecoveryStatus status) const;
    
    // Cleanup operations
    Result<void> cleanup_expired_backups();

private:
    // Helper methods
    std::string generate_backup_id() const;
    std::string generate_recovery_id() const;
    Result<std::string> perform_backup(const BackupConfig& config, const std::string& backup_id);
    Result<std::string> perform_recovery(const RecoveryConfig& config, const std::string& recovery_id);
    Result<void> compress_backup_data(const std::string& source_path, const std::string& target_path, 
                                    const std::string& method) const;
    Result<void> decompress_backup_data(const std::string& source_path, const std::string& target_path, 
                                      const std::string& method) const;
    Result<void> encrypt_backup_data(const std::string& source_path, const std::string& target_path,
                                   const std::string& key) const;
    Result<void> decrypt_backup_data(const std::string& source_path, const std::string& target_path,
                                   const std::string& key) const;
    std::string calculate_checksum(const std::string& file_path) const;
    void cleanup_temp_files(const std::string& backup_path) const;
};

} // namespace jadevectordb

#endif // JADEVECTORDB_BACKUP_SERVICE_H