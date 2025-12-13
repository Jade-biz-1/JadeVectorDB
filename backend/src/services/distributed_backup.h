#pragma once

#include "lib/result.h"
#include "lib/logging.h"
#include "sharding_service.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>

namespace jadevectordb {

struct BackupMetadata {
    std::string backup_id;
    std::string backup_name;
    int64_t timestamp;
    std::string backup_type; // "full", "incremental"
    int64_t size_bytes;
    std::vector<std::string> included_shards;
    std::string status;
    std::map<std::string, std::string> metadata;
};

struct RestoreStatus {
    std::string restore_id;
    std::string backup_id;
    int64_t started_at;
    int64_t completed_at;
    int64_t total_bytes;
    int64_t restored_bytes;
    std::string status;
    std::string error_message;
};

class DistributedBackupService {
public:
    DistributedBackupService();
    ~DistributedBackupService();
    
    bool initialize(std::shared_ptr<ShardingService> sharding_service);
    Result<bool> start();
    void stop();
    
    // Backup operations
    Result<std::string> create_full_backup(const std::string& backup_name);
    Result<std::string> create_incremental_backup(const std::string& base_backup_id);
    Result<std::string> create_snapshot();
    
    // Restore operations
    Result<std::string> restore_from_backup(const std::string& backup_id);
    Result<std::string> restore_to_point_in_time(int64_t timestamp);
    
    // Verification
    Result<bool> verify_backup(const std::string& backup_id);
    
    // Management
    std::vector<BackupMetadata> list_backups();
    Result<BackupMetadata> get_backup_metadata(const std::string& backup_id);
    Result<bool> delete_backup(const std::string& backup_id);
    Result<RestoreStatus> get_restore_status(const std::string& restore_id);
    
private:
    std::shared_ptr<ShardingService> sharding_service_;
    std::map<std::string, BackupMetadata> backups_;
    std::map<std::string, RestoreStatus> restores_;
    mutable std::mutex mutex_;
    std::atomic<bool> running_{false};
    std::shared_ptr<logging::Logger> logger_;
    
    std::string generate_backup_id();
    std::string generate_restore_id();
};

} // namespace jadevectordb
